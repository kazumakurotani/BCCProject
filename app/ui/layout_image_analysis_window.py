# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'layout_image_analysis_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow_for_ImageAnalysis(object):
    def setupUi(self, MainWindow_for_ImageAnalysis):
        MainWindow_for_ImageAnalysis.setObjectName("MainWindow_for_ImageAnalysis")
        MainWindow_for_ImageAnalysis.resize(600, 500)
        MainWindow_for_ImageAnalysis.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"background-color: rgb(240, 240, 240);")
        self.centralwidget = QtWidgets.QWidget(MainWindow_for_ImageAnalysis)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_for_selectDirectory = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_for_selectDirectory.sizePolicy().hasHeightForWidth())
        self.pushButton_for_selectDirectory.setSizePolicy(sizePolicy)
        self.pushButton_for_selectDirectory.setStyleSheet("QPushButton {\n"
"    font: 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(0, 0, 0);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 1px solid rgb(0, 0, 0);\n"
"    border-radius: 1px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    font: 75 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(85, 170, 255);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 2px solid rgb(85, 170, 255);\n"
"}")
        self.pushButton_for_selectDirectory.setObjectName("pushButton_for_selectDirectory")
        self.verticalLayout_3.addWidget(self.pushButton_for_selectDirectory)
        self.treeView_for_selectDirectoly = QtWidgets.QTreeView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeView_for_selectDirectoly.sizePolicy().hasHeightForWidth())
        self.treeView_for_selectDirectoly.setSizePolicy(sizePolicy)
        self.treeView_for_selectDirectoly.setTextElideMode(QtCore.Qt.ElideLeft)
        self.treeView_for_selectDirectoly.setIndentation(100)
        self.treeView_for_selectDirectoly.setObjectName("treeView_for_selectDirectoly")
        self.treeView_for_selectDirectoly.header().setMinimumSectionSize(50)
        self.verticalLayout_3.addWidget(self.treeView_for_selectDirectoly)
        self.pushButton_for_executeAnalysis = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_for_executeAnalysis.sizePolicy().hasHeightForWidth())
        self.pushButton_for_executeAnalysis.setSizePolicy(sizePolicy)
        self.pushButton_for_executeAnalysis.setStyleSheet("QPushButton {\n"
"    font: 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(0, 0, 0);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 1px solid rgb(0, 0, 0);\n"
"    border-radius: 1px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    font: 75 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(85, 170, 255);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 2px solid rgb(85, 170, 255);\n"
"}")
        self.pushButton_for_executeAnalysis.setObjectName("pushButton_for_executeAnalysis")
        self.verticalLayout_3.addWidget(self.pushButton_for_executeAnalysis)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.comboBox_for_selectImage = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_for_selectImage.sizePolicy().hasHeightForWidth())
        self.comboBox_for_selectImage.setSizePolicy(sizePolicy)
        self.comboBox_for_selectImage.setObjectName("comboBox_for_selectImage")
        self.verticalLayout_2.addWidget(self.comboBox_for_selectImage)
        self.comboBox_for_selectFeature = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_for_selectFeature.sizePolicy().hasHeightForWidth())
        self.comboBox_for_selectFeature.setSizePolicy(sizePolicy)
        self.comboBox_for_selectFeature.setObjectName("comboBox_for_selectFeature")
        self.verticalLayout_2.addWidget(self.comboBox_for_selectFeature)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.pushButton_for_viewDiagram = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_for_viewDiagram.sizePolicy().hasHeightForWidth())
        self.pushButton_for_viewDiagram.setSizePolicy(sizePolicy)
        self.pushButton_for_viewDiagram.setStyleSheet("QPushButton {\n"
"    font: 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(0, 0, 0);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 1px solid rgb(0, 0, 0);\n"
"    border-radius: 1px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    font: 75 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(85, 170, 255);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 2px solid rgb(85, 170, 255);\n"
"}")
        self.pushButton_for_viewDiagram.setObjectName("pushButton_for_viewDiagram")
        self.horizontalLayout_2.addWidget(self.pushButton_for_viewDiagram)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget_for_viewDiagram = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_for_viewDiagram.setStyleSheet("    background-color: rgb(255, 255, 255);\n"
"")
        self.tabWidget_for_viewDiagram.setObjectName("tabWidget_for_viewDiagram")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.graphicsView_tab_1 = QtWidgets.QGraphicsView(self.tab_1)
        self.graphicsView_tab_1.setObjectName("graphicsView_tab_1")
        self.gridLayout_2.addWidget(self.graphicsView_tab_1, 0, 0, 1, 1)
        self.tableView_tab_1 = QtWidgets.QTableView(self.tab_1)
        self.tableView_tab_1.setObjectName("tableView_tab_1")
        self.gridLayout_2.addWidget(self.tableView_tab_1, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.graphicsView_tab_2 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_tab_2.setObjectName("graphicsView_tab_2")
        self.gridLayout_3.addWidget(self.graphicsView_tab_2, 0, 0, 1, 1)
        self.tableView_tab_2 = QtWidgets.QTableView(self.tab_2)
        self.tableView_tab_2.setObjectName("tableView_tab_2")
        self.gridLayout_3.addWidget(self.tableView_tab_2, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.graphicsView_tab_3 = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsView_tab_3.setObjectName("graphicsView_tab_3")
        self.gridLayout_4.addWidget(self.graphicsView_tab_3, 0, 0, 1, 1)
        self.tableView_tab_3 = QtWidgets.QTableView(self.tab_3)
        self.tableView_tab_3.setObjectName("tableView_tab_3")
        self.gridLayout_4.addWidget(self.tableView_tab_3, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.graphicsView_tab_4 = QtWidgets.QGraphicsView(self.tab_4)
        self.graphicsView_tab_4.setObjectName("graphicsView_tab_4")
        self.gridLayout_5.addWidget(self.graphicsView_tab_4, 0, 0, 1, 1)
        self.tableView_tab_4 = QtWidgets.QTableView(self.tab_4)
        self.tableView_tab_4.setObjectName("tableView_tab_4")
        self.gridLayout_5.addWidget(self.tableView_tab_4, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.graphicsView_tab_5 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_tab_5.setObjectName("graphicsView_tab_5")
        self.gridLayout_6.addWidget(self.graphicsView_tab_5, 0, 0, 1, 1)
        self.tableView_tab_5 = QtWidgets.QTableView(self.tab_5)
        self.tableView_tab_5.setObjectName("tableView_tab_5")
        self.gridLayout_6.addWidget(self.tableView_tab_5, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_6)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.graphicsView_tab_6 = QtWidgets.QGraphicsView(self.tab_6)
        self.graphicsView_tab_6.setObjectName("graphicsView_tab_6")
        self.gridLayout_7.addWidget(self.graphicsView_tab_6, 0, 0, 1, 1)
        self.tableView_tab_6 = QtWidgets.QTableView(self.tab_6)
        self.tableView_tab_6.setObjectName("tableView_tab_6")
        self.gridLayout_7.addWidget(self.tableView_tab_6, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.tableView_tab_7 = QtWidgets.QTableView(self.tab_7)
        self.tableView_tab_7.setGeometry(QtCore.QRect(10, 200, 256, 192))
        self.tableView_tab_7.setObjectName("tableView_tab_7")
        self.graphicsView_tab_7 = QtWidgets.QGraphicsView(self.tab_7)
        self.graphicsView_tab_7.setGeometry(QtCore.QRect(10, 0, 256, 192))
        self.graphicsView_tab_7.setObjectName("graphicsView_tab_7")
        self.tabWidget_for_viewDiagram.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tab_8)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.graphicsView_tab_8 = QtWidgets.QGraphicsView(self.tab_8)
        self.graphicsView_tab_8.setObjectName("graphicsView_tab_8")
        self.gridLayout_8.addWidget(self.graphicsView_tab_8, 0, 0, 1, 1)
        self.tableView_tab8 = QtWidgets.QTableView(self.tab_8)
        self.tableView_tab8.setObjectName("tableView_tab8")
        self.gridLayout_8.addWidget(self.tableView_tab8, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_8, "")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.tab_9)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.graphicsView_tab_9 = QtWidgets.QGraphicsView(self.tab_9)
        self.graphicsView_tab_9.setObjectName("graphicsView_tab_9")
        self.gridLayout_9.addWidget(self.graphicsView_tab_9, 0, 0, 1, 1)
        self.tableView_tab_9 = QtWidgets.QTableView(self.tab_9)
        self.tableView_tab_9.setObjectName("tableView_tab_9")
        self.gridLayout_9.addWidget(self.tableView_tab_9, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_9, "")
        self.tab_10 = QtWidgets.QWidget()
        self.tab_10.setObjectName("tab_10")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.tab_10)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.graphicsView_tab_10 = QtWidgets.QGraphicsView(self.tab_10)
        self.graphicsView_tab_10.setObjectName("graphicsView_tab_10")
        self.gridLayout_10.addWidget(self.graphicsView_tab_10, 0, 0, 1, 1)
        self.tableView_tab_10 = QtWidgets.QTableView(self.tab_10)
        self.tableView_tab_10.setObjectName("tableView_tab_10")
        self.gridLayout_10.addWidget(self.tableView_tab_10, 1, 0, 1, 1)
        self.tabWidget_for_viewDiagram.addTab(self.tab_10, "")
        self.tab_11 = QtWidgets.QWidget()
        self.tab_11.setObjectName("tab_11")
        self.tabWidget_for_viewDiagram.addTab(self.tab_11, "")
        self.tab_12 = QtWidgets.QWidget()
        self.tab_12.setObjectName("tab_12")
        self.tabWidget_for_viewDiagram.addTab(self.tab_12, "")
        self.verticalLayout.addWidget(self.tabWidget_for_viewDiagram)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.pushButton_for_saveResults = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_for_saveResults.sizePolicy().hasHeightForWidth())
        self.pushButton_for_saveResults.setSizePolicy(sizePolicy)
        self.pushButton_for_saveResults.setStyleSheet("QPushButton {\n"
"    font: 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(0, 0, 0);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 1px solid rgb(0, 0, 0);\n"
"    border-radius: 1px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    font: 75 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(85, 170, 255);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 2px solid rgb(85, 170, 255);\n"
"}")
        self.pushButton_for_saveResults.setObjectName("pushButton_for_saveResults")
        self.horizontalLayout.addWidget(self.pushButton_for_saveResults)
        self.pushButton_for_closeWindow = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_for_closeWindow.sizePolicy().hasHeightForWidth())
        self.pushButton_for_closeWindow.setSizePolicy(sizePolicy)
        self.pushButton_for_closeWindow.setStyleSheet("QPushButton {\n"
"    font: 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(0, 0, 0);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 1px solid rgb(0, 0, 0);\n"
"    border-radius: 1px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    font: 75 11pt \"Segoe UI\";\n"
"    font-weight: bold;\n"
"    color: rgb(85, 170, 255);\n"
"    background-color: rgb(255, 255, 255);\n"
"    border: 2px solid rgb(85, 170, 255);\n"
"}")
        self.pushButton_for_closeWindow.setFlat(False)
        self.pushButton_for_closeWindow.setObjectName("pushButton_for_closeWindow")
        self.horizontalLayout.addWidget(self.pushButton_for_closeWindow)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        MainWindow_for_ImageAnalysis.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow_for_ImageAnalysis)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow_for_ImageAnalysis.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow_for_ImageAnalysis)
        self.statusbar.setObjectName("statusbar")
        MainWindow_for_ImageAnalysis.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow_for_ImageAnalysis)
        self.tabWidget_for_viewDiagram.setCurrentIndex(0)
        self.pushButton_for_selectDirectory.clicked.connect(MainWindow_for_ImageAnalysis.select_directory) # type: ignore
        self.pushButton_for_executeAnalysis.clicked.connect(MainWindow_for_ImageAnalysis.execute) # type: ignore
        self.pushButton_for_saveResults.clicked.connect(MainWindow_for_ImageAnalysis.save_results) # type: ignore
        self.pushButton_for_closeWindow.clicked.connect(MainWindow_for_ImageAnalysis.close) # type: ignore
        self.pushButton_for_viewDiagram.clicked.connect(MainWindow_for_ImageAnalysis.view_diagram) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow_for_ImageAnalysis)

    def retranslateUi(self, MainWindow_for_ImageAnalysis):
        _translate = QtCore.QCoreApplication.translate
        MainWindow_for_ImageAnalysis.setWindowTitle(_translate("MainWindow_for_ImageAnalysis", "MainWindow"))
        self.pushButton_for_selectDirectory.setText(_translate("MainWindow_for_ImageAnalysis", "select directory"))
        self.pushButton_for_executeAnalysis.setText(_translate("MainWindow_for_ImageAnalysis", "execute"))
        self.pushButton_for_viewDiagram.setText(_translate("MainWindow_for_ImageAnalysis", "  view  "))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_1), _translate("MainWindow_for_ImageAnalysis", "1"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_2), _translate("MainWindow_for_ImageAnalysis", "2"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_3), _translate("MainWindow_for_ImageAnalysis", "3"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_4), _translate("MainWindow_for_ImageAnalysis", "4"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_5), _translate("MainWindow_for_ImageAnalysis", "5"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_6), _translate("MainWindow_for_ImageAnalysis", "6"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_7), _translate("MainWindow_for_ImageAnalysis", "7"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_8), _translate("MainWindow_for_ImageAnalysis", "8"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_9), _translate("MainWindow_for_ImageAnalysis", "9"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_10), _translate("MainWindow_for_ImageAnalysis", "10"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_11), _translate("MainWindow_for_ImageAnalysis", "12"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_12), _translate("MainWindow_for_ImageAnalysis", "13"))
        self.pushButton_for_saveResults.setText(_translate("MainWindow_for_ImageAnalysis", "SAVE"))
        self.pushButton_for_closeWindow.setText(_translate("MainWindow_for_ImageAnalysis", "CLOSE"))
