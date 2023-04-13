import sys
import traceback

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from mrfi_gui.mainwindow import Ui_MainWindow
from mrfi.mrfi import ConfigTree, read_config

class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.actionLoadConfig.triggered.connect(self.loadConfig)
        self.initUI()
    
    def initUI(self):
        model = QStandardItemModel(self)
        self.label.setText('Load a tree config file')

        self.baseitem = QStandardItem('model')
        self.baseitem.setEditable(False)
        
        model.appendRow(self.baseitem)

        self.treeView.setModel(model)
        self.treeView.expandToDepth(1)
    
    def loadConfig(self):
        filename = QFileDialog.getOpenFileName(self, 
                'Load config file', './configs', 'yaml (*.yml;*.yaml);;xml (*.xml);;json (*.json);;All (*.*)')[0]
        try:
            config_root = read_config(filename)
            configtree = ConfigTree(config_root, None)
        except Exception as e:
            QMessageBox.critical(self, 'Error on loading config', traceback.format_exc())
            return
        
        for k, v in configtree.raw_dict.items():
            item = QStandardItem(k)
            item.setEditable(False)
            self.__load_subconfig(item, v)
            self.baseitem.appendRow(item)

    def __load_subconfig(self, itemnode, configtree):
        if configtree.raw_dict is not None:
            for k, v in configtree.raw_dict.items():
                item = QStandardItem(k)
                item.setEditable(False)
                if isinstance(v, ConfigTree):
                    self.__load_subconfig(item, v)
                itemnode.appendRow(item)
        else:
            for k, v in enumerate(configtree.raw_list):
                item = QStandardItem(str(k))
                item.setEditable(False)
                if isinstance(v, ConfigTree):
                    self.__load_subconfig(item, v)
                itemnode.appendRow(item)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainForm()
    window.show()
    sys.exit(app.exec_())
