import sys
import traceback

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from mrfi_gui.mainwindow import Ui_MainWindow
from mrfi.mrfi import ConfigTree, _read_config, ConfigTreeNodeType

class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.actionLoadConfig.triggered.connect(self.loadConfig)
        self.treeView.clicked.connect(self.treeviewClicked)
        self.initUI()
    
    def treeviewClicked(self, index):
        selitem = self.model.itemFromIndex(index)

        if id(selitem) in self.item2config_detailmodel:
            cfg, detailmodel = self.item2config_detailmodel[id(selitem)]
            self.treeView_2.setModel(detailmodel)
            self.treeView_2.expandToDepth(1)
            self.label.setText(cfg.name)
        else:
            self.treeView_2.setModel(self.emptytreeview2model)
            self.label.setText('N/A')

    def initUI(self):
        self.model = QStandardItemModel(self)
        headerlabels = ['module', 'act in', 'act out', 'weights', 'obs', 'obs pre']
        self.model.setHorizontalHeaderLabels(headerlabels)
        self.treeView.setModel(self.model)
        for i in range(6):
            if i == 0:
                self.treeView.setColumnWidth(i, 180)
            else:
                self.treeView.setColumnWidth(i, 80)

        self.emptytreeview2model = QStandardItemModel(self)
        self.emptytreeview2model.setHorizontalHeaderLabels(['item', 'value'])
        self.treeView_2.setModel(self.emptytreeview2model)
        self.treeView_2.setColumnWidth(0, 200)
        self.treeView_2.setColumnWidth(1, 80)
    
    def loadConfig(self):
        filename = QFileDialog.getOpenFileName(self, 
                'Load config file', './detailconfigs', 'yaml (*.yml;*.yaml);;xml (*.xml);;json (*.json);;All (*.*)')[0]
        if filename == '': return
        try:
            config_root = _read_config(filename)
            configtree = ConfigTree(config_root, None)
        except Exception as e:
            QMessageBox.critical(self, 'Error on loading config', traceback.format_exc())
            return
        self.label_2.setText(filename.split('/')[-1])
        self.label.setText('Choose an item to edit')
        
        self.model = QStandardItemModel(self)
        self.item2config_detailmodel = {}
        self.load_subconfig(self.model, configtree)
        self.treeView.setModel(self.model)
        self.treeView.expandToDepth(0)
        
        headerlabels = ['module', 'act in', 'act out', 'weights', 'obs', 'obs pre']
        self.model.setHorizontalHeaderLabels(headerlabels)
    
    def load_subdetail_config(self, basenode, configtree, editable = False):
        for k, v in configtree.raw_dict.items():
            item = QStandardItem(str(k))
            item.setEditable(editable)
            items = [item]
                
            if isinstance(v, ConfigTree):
                if v.nodetype == ConfigTreeNodeType.FI_STAGE:
                    methoditem = QStandardItem(str(v.method))
                    items.append(methoditem)
                    self.load_subdetail_config(item, v.args, True)
                else:
                    self.load_subdetail_config(item, v)
            else:
                if k == 'enabled':
                    valueitem = QStandardItem('')
                    valueitem.setEditable(False)
                    valueitem.setCheckState(bool(v)*2)
                    valueitem.setCheckable(True)
                else:
                    valueitem = QStandardItem(str(v))
                items.append(valueitem)
            basenode.appendRow(items)

    def load_detail_model(self, configtree):
        model = QStandardItemModel(self)
        model.setHorizontalHeaderLabels(['item', 'value'])
        self.load_subdetail_config(model, configtree)
        return model
    
    def load_subconfig(self, basenode, configtree):
        cols = [QStandardItem(configtree.name.split('.')[-1])]
        self.item2config_detailmodel[id(cols[0])] = (configtree, self.emptytreeview2model)
        
        colprops = ['activation', 'activation_out', 'weights', 'observers', 'observers_pre']

        for colprop in colprops:
            if configtree.hasattr(colprop):
                qitem = QStandardItem(str(len(configtree[colprop])))
                self.item2config_detailmodel[id(qitem)] = (configtree[colprop], 
                                                    self.load_detail_model(configtree[colprop]))
            else:
                qitem = QStandardItem('N/A')

            qitem.setEditable(False)
            cols.append(qitem)

        basenode.appendRow(cols)

        if configtree.hasattr('sub_modules'):
            for v in configtree.sub_modules:
                self.load_subconfig(cols[0], v)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainForm()
    window.show()
    sys.exit(app.exec_())
