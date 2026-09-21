from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import json5
import pytest
from PIL import Image

from qwen_studio.comfy import ComfyClient, EDIT_MODEL
from qwen_studio.desktop import menu_content
from qwen_studio.editing import workflow
from qwen_studio.gallery import Gallery
from qwen_studio.imaging import load_image, outpaint_input, transform
from qwen_studio.jobs import JobRunner
from tests.conftest import FakeComfy, write_png
from tests.test_server import _wait_job


def test_project_create_rename_move_delete_persists(studio):
    client, _, gallery = studio
    project = client.post('/api/projects', json={'name': 'Characters / chapter 1'}).json()
    assert client.patch('/api/projects/'+project['id'], json={'name': 'Characters'}).status_code == 200
    assert Gallery(gallery.root).projects() == [{'id': project['id'], 'name': 'Characters'}]
    image = write_png(gallery.root / 'hero.png')
    image.with_suffix('.json').write_text('{"prompt":"A hero","seed":9223372036854775807}')
    response = client.post('/api/images/move', json={'filename':image.name,'project':project['id']})
    assert response.status_code == 200
    filename = response.json()['filename']
    assert not image.exists()
    assert gallery.resolve(filename).with_suffix('.json').is_file()
    record = client.get('/api/images').json()['images'][0]
    assert record['project'] == project['id'] and record['seed'] == '9223372036854775807'
    assert client.get(record['url']).status_code == 200
    assert client.delete('/api/images/'+filename).status_code == 200
    assert client.get(record['url']).status_code == 404
    assert gallery.list_images() == []


def test_project_collision_never_overwrites(tmp_path):
    gallery = Gallery(tmp_path)
    project = gallery.save_project('Test')['id']
    write_png(tmp_path/'same.png')
    target = gallery.project_dir(project)/'same.png'
    target.write_bytes(b'keep this')
    moved = gallery.move('same.png',project)
    assert Path(moved).name != 'same.png'
    assert target.read_bytes() == b'keep this'


@pytest.mark.parametrize('filename', ['../secret.png','projects/../secret.png','projects/bad/a.png','projects/'+'a'*32+'/../a.png','/tmp/x.png','sub/x.png','projects\\test\\x.png'])
def test_nested_paths_rejected(tmp_path, filename):
    with pytest.raises((ValueError,FileNotFoundError)):
        Gallery(tmp_path).resolve(filename)


def test_project_symlinks_rejected(tmp_path):
    gallery = Gallery(tmp_path)
    external = tmp_path/'external'; external.mkdir()
    (tmp_path/'projects').symlink_to(external)
    with pytest.raises(ValueError): gallery.save_project('Escape')
    assert gallery.projects() == []


def test_generate_directly_into_project(studio):
    client, _, gallery = studio
    project = client.post('/api/projects',json={'name':'New work'}).json()['id']
    response = client.post('/api/generate',json={'prompt':'cube','project':project})
    job = _wait_job(client,response.json()['id'])
    assert job['status']=='done'
    assert job['filename'].startswith('projects/'+project+'/')
    assert gallery.resolve(job['filename']).is_file()


def test_import_and_local_upscale_keep_original(studio):
    client, _, gallery = studio
    buffer = io.BytesIO(); Image.new('RGB',(64,64),'red').save(buffer,format='PNG')
    imported = client.post('/api/import',content=buffer.getvalue()).json()['filename']
    before = gallery.resolve(imported).read_bytes()
    response = client.post('/api/generate',json={'operation':'upscale','sources':[imported], 'width':128,'height':128})
    result = _wait_job(client,response.json()['id'])
    assert result['status']=='done'
    assert load_image(gallery.resolve(result['filename'])).size == (128,128)
    assert gallery.resolve(imported).read_bytes()==before
    assert client.post('/api/import',content=b'not an image').status_code==400
    assert client.post('/api/generate',json={'operation':'crop','width':0,'sources':[imported]}).status_code==400


def test_queued_sources_survive_move_and_delete(tmp_path):
    async def run():
        source = write_png(tmp_path/'original.png',width=64,height=64)
        gallery = Gallery(tmp_path)
        runner = JobRunner(FakeComfy(tmp_path),gallery)
        job = runner.enqueue('', '', 128,128,30,1,operation='upscale',sources=[source.name])
        workspace = Path(job.workspace)
        gallery.delete(source.name)
        runner.start(); await asyncio.wait_for(runner._queue.join(),3); await runner.stop()
        assert job.status=='done'
        assert not workspace.exists()
    asyncio.run(run())


def test_local_transform_geometry(tmp_path):
    source = tmp_path/'source.png'; Image.new('RGB',(128,64),'red').save(source)
    assert load_image(transform(source,tmp_path,'crop',64,64)).size==(64,64)
    fit = load_image(transform(source,tmp_path,'fit',128,128))
    assert fit.getpixel((0,0))==(18,20,24)
    assert fit.getpixel((64,64))==(255,0,0)
    with pytest.raises(ValueError): transform(source,tmp_path,'upscale',128,128)


def test_outpaint_mask_keeps_source(tmp_path):
    source = tmp_path/'source.png'; Image.new('RGB',(64,64),'red').save(source)
    target = tmp_path/'padded.png'
    assert outpaint_input(source,target,128,64)==(32,0)
    with Image.open(target) as im:
        assert im.getpixel((0,0))[3]==0
        assert im.getpixel((32,0))==(255,0,0,255)
    with pytest.raises(ValueError): outpaint_input(source,target,32,64)


def test_reference_workflow_and_mask_graph():
    graph = workflow(ComfyClient(),'new pose','',512,512,30,9,['a.png','b.png'],'edit')
    assert graph['1']['inputs']['unet_name']==EDIT_MODEL
    assert graph['5']['class_type']=='TextEncodeQwenImageEditPlus'
    assert graph['5']['inputs']['image2']==['12',0]
    assert graph['7']['class_type']=='EmptySD3LatentImage'
    extended = workflow(ComfyClient(),'extend','',1024,512,30,9,['a.png'],'extend')
    assert extended['7']['inputs']['mask']==['11',1]


def test_menu_install_idempotent_preserves_entries_and_comments():
    original = '{\n// private customization\n"custom":{"label":"Custom"},"custom.existing":{"action":"my-app"}\n}\n'
    installed = menu_content(original,'"/home/user/.local/bin/qwen-studio-open"')
    twice = menu_content(installed,'"/home/user/.local/bin/qwen-studio-open"')
    assert installed==twice
    parsed=json5.loads(twice)
    assert parsed['custom.existing']['action']=='my-app'
    assert '// private customization' in twice
    removed=menu_content(twice,'',remove=True)
    assert json5.loads(removed)==json5.loads(original)


def test_menu_new_install():
    parsed=json5.loads(menu_content('{}','qwen-studio-open'))
    assert parsed['custom.qwen-studio']['parent']=='custom'


def test_remove_menu_block_when_user_added_item_after_it():
    text = menu_content('{}', 'qwen-studio-open')
    text = text[:text.rfind('}')] + ',"later":{"label":"User item"}}'
    result = menu_content(text, '', remove=True)
    assert json5.loads(result) == {'later': {'label': 'User item'}}


def test_menu_preserves_trailing_line_comment():
    original = '{"custom": {"label":"Custom"}\n // END another-app\n}\n'
    updated = menu_content(original, 'qwen-studio-open')
    assert json5.loads(updated)['custom.qwen-studio']['action'] == 'qwen-studio-open'
    assert '// END another-app' in updated
