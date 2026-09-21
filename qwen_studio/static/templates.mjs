export const STYLE_TEMPLATES = [
  { id: 'anime', name: 'Anime', mark: '✦', color: '#ec90b1', description: 'Expressive drawing, crisp contours, luminous color.',
    prompt: 'Anime illustration with expressive shapes, precise linework, layered cel shading, luminous colors, detailed hand-painted backgrounds, and a thoughtfully composed frame.',
    negative: 'photographic rendering, live-action, muddy colors, inconsistent linework' },
  { id: 'cartoon', name: 'Cartoon', mark: '◒', color: '#f0c85d', description: 'Playful shapes and bold, readable silhouettes.',
    prompt: 'A playful 2D cartoon illustration, appealing exaggerated shapes, bold clean outlines, clear silhouettes, flat vibrant colors, simple deliberate shading, and expressive visual storytelling.',
    negative: 'photorealism, gritty surface detail, stiff expressions, cluttered silhouettes' },
  { id: 'realistic', name: 'Realistic', mark: '◉', color: '#a0c4ab', description: 'Natural light, tactile materials, believable detail.',
    prompt: 'A photorealistic photograph with natural lighting, believable proportions, physically accurate materials, fine surface texture, subtle color grading, realistic shadows, and a considered photographic composition.',
    negative: 'illustration, cartoon, plastic textures, excessive skin smoothing, artificial HDR halos' },
  { id: 'medieval', name: 'Medieval', mark: '♜', color: '#c4ab7d', description: 'Weathered stone, candlelight, historical atmosphere.',
    prompt: 'A richly detailed medieval scene with period-appropriate clothing, hand-forged objects, weathered stone and timber, tactile linen and leather, muted earth tones, atmospheric natural light or candlelight, and a historical storybook sensibility.',
    negative: 'modern technology, electric lighting, contemporary clothing, plastic, modern signage' },
  { id: 'inkpen', name: 'Ink pen', mark: '╱', color: '#c6c2b8', description: 'Fine nibs, crosshatching, warm paper.',
    prompt: 'A black ink pen drawing on warm ivory paper, confident fine nib linework, delicate crosshatching and stippling, varied line weight, expressive contours, carefully preserved negative space, and no color fills.',
    negative: 'full color, airbrushed gradients, photographic rendering, thick paint, blurred lines' },
  { id: 'stylized', name: 'Stylized', mark: '◆', color: '#91b2df', description: 'Sculpted forms with a deliberate color palette.',
    prompt: 'A stylized editorial illustration with sculptural simplified forms, elegant shape language, an intentional limited color palette, exaggerated proportions, soft graphic shadows, and a strong focal point.',
    negative: 'photorealism, visual clutter, inconsistent rendering styles, excessive microdetail' },
  { id: 'watercolor', name: 'Watercolor', mark: '≈', color: '#91c7cc', description: 'Translucent washes and soft pigment edges.',
    prompt: 'A delicate watercolor painting on textured cotton paper, transparent layered washes, subtle pigment blooms, soft edges balanced with a few precise details, luminous reserved whites, and a light hand-painted atmosphere.',
    negative: 'plastic surfaces, hard vector outlines, digital gloss, opaque flat fills' },
  { id: 'oil', name: 'Oil painting', mark: '▰', color: '#cf9272', description: 'Visible brushwork and rich painted light.',
    prompt: 'An expressive oil painting with visible brushwork, layered pigment, rich restrained colors, luminous highlights, deep nuanced shadows, a tactile canvas surface, and a balanced painterly composition.',
    negative: 'vector art, flat digital fills, photographic sharpness everywhere, plastic shine' },
  { id: 'pixel', name: 'Pixel art', mark: '▦', color: '#b1c775', description: 'Crisp clusters and a small, cohesive palette.',
    prompt: 'Carefully crafted pixel art, crisp square pixels at a consistent scale, intentional pixel clusters, a small cohesive color palette, readable silhouettes, selective dithering, and no anti-aliasing.',
    negative: 'smooth gradients, blurry edges, mixed pixel scales, photorealism, anti-aliasing' },
  { id: 'cinematic', name: 'Cinematic', mark: '▱', color: '#d6a267', description: 'Motivated light and a film-still composition.',
    prompt: 'A cinematic film still with motivated lighting, a strong visual focal point, layered foreground and background, subtle film grain, controlled contrast, atmospheric depth, and a restrained film color grade.',
    negative: 'flat lighting, oversharpening, excessive lens flare, text overlays, letterbox borders' },
  { id: 'cyberpunk', name: 'Cyberpunk', mark: 'ϟ', color: '#78d5c4', description: 'Neon reflections and dense future-city detail.',
    prompt: 'A cinematic cyberpunk interpretation, densely layered futuristic design, worn industrial materials, neon accent lighting, rain-slick reflections, atmospheric haze, deep shadows, and a tightly controlled teal and magenta palette.',
    negative: 'pastoral scenery, pristine empty environments, flat daylight, excessive neon bloom' },
  { id: 'clay', name: 'Clay miniature', mark: '●', color: '#dca9a0', description: 'Handmade shapes and a miniature-set feel.',
    prompt: 'A handmade clay miniature photographed on a carefully built miniature set, rounded sculpted forms, subtle fingerprints and tool marks, tactile matte surfaces, soft studio lighting, and charming stop-motion craftsmanship.',
    negative: 'glossy plastic, photorealistic flesh, razor-sharp mechanical edges, flat vector art' },
];

export function composePrompt(subject, style) {
  return [subject.trim(), style.trim()].filter(Boolean).join('\n\n');
}

export function generationSettings(draft) {
  return {
    prompt: composePrompt(draft.prompt, draft.style_prompt || ''),
    negative_prompt: [draft.negative_prompt.trim(), (draft.style_negative || '').trim()].filter(Boolean).join(', '),
    width: draft.width, height: draft.height, steps: draft.steps, seed: draft.seed,
  };
}
