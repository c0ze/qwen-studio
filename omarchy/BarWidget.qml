import QtQuick
import qs.Ui

BarWidget {
  id: root
  moduleName: "c0ze.qwen-studio"
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: "✦"
    horizontalMargin: 8
    onPressed: function(button) {
      if (root.bar && button === Qt.LeftButton) root.bar.run("qwen-studio-open")
    }
  }
}
