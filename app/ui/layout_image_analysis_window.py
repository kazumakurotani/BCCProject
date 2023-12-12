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
        MainWindow_for_ImageAnalysis.resize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow_for_ImageAnalysis)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
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
        self.verticalLayout_2.addWidget(self.pushButton_for_selectDirectory)
        self.treeView_for_selectDirectoly = QtWidgets.QTreeView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeView_for_selectDirectoly.sizePolicy().hasHeightForWidth())
        self.treeView_for_selectDirectoly.setSizePolicy(sizePolicy)
        self.treeView_for_selectDirectoly.setObjectName("treeView_for_selectDirectoly")
        self.verticalLayout_2.addWidget(self.treeView_for_selectDirectoly)
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
        self.verticalLayout_2.addWidget(self.pushButton_for_executeAnalysis)
        self.pushButton_for_saveResults = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
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
        self.verticalLayout_2.addWidget(self.pushButton_for_saveResults)
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
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget_for_viewDiagram = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_for_viewDiagram.setStyleSheet("    background-color: rgb(255, 255, 255);\n"
"    border: 0.5px solid rgb(0, 0, 0);\n"
"    border-radius: 1px;\n"
"")
        self.tabWidget_for_viewDiagram.setObjectName("tabWidget_for_viewDiagram")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget_for_viewDiagram.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget_for_viewDiagram.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget_for_viewDiagram.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget_for_viewDiagram.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.tabWidget_for_viewDiagram.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.tabWidget_for_viewDiagram.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.tabWidget_for_viewDiagram.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.tabWidget_for_viewDiagram.addTab(self.tab_8, "")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.tabWidget_for_viewDiagram.addTab(self.tab_9, "")
        self.tab_10 = QtWidgets.QWidget()
        self.tab_10.setObjectName("tab_10")
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
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
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
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        MainWindow_for_ImageAnalysis.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow_for_ImageAnalysis)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 22))
        self.menubar.setObjectName("menubar")
        MainWindow_for_ImageAnalysis.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow_for_ImageAnalysis)
        self.statusbar.setObjectName("statusbar")
        MainWindow_for_ImageAnalysis.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow_for_ImageAnalysis)
        self.tabWidget_for_viewDiagram.setCurrentIndex(11)
        self.pushButton_for_selectDirectory.clicked.connect(MainWindow_for_ImageAnalysis.select_directory) # type: ignore
        self.pushButton_for_executeAnalysis.clicked.connect(MainWindow_for_ImageAnalysis.execute) # type: ignore
        self.pushButton_for_saveResults.clicked.connect(MainWindow_for_ImageAnalysis.save_results) # type: ignore
        self.pushButton_for_closeWindow.clicked.connect(MainWindow_for_ImageAnalysis.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow_for_ImageAnalysis)

    def retranslateUi(self, MainWindow_for_ImageAnalysis):
        _translate = QtCore.QCoreApplication.translate
        MainWindow_for_ImageAnalysis.setWindowTitle(_translate("MainWindow_for_ImageAnalysis", "MainWindow"))
        self.pushButton_for_selectDirectory.setText(_translate("MainWindow_for_ImageAnalysis", "select directory"))
        self.pushButton_for_executeAnalysis.setText(_translate("MainWindow_for_ImageAnalysis", "execute"))
        self.pushButton_for_saveResults.setText(_translate("MainWindow_for_ImageAnalysis", "save"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab), _translate("MainWindow_for_ImageAnalysis", "Tab 1"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_2), _translate("MainWindow_for_ImageAnalysis", "Tab 2"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_3), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_4), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_5), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_6), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_7), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_8), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_9), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_10), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_11), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.tabWidget_for_viewDiagram.setTabText(self.tabWidget_for_viewDiagram.indexOf(self.tab_12), _translate("MainWindow_for_ImageAnalysis", "ページ"))
        self.pushButton_for_closeWindow.setText(_translate("MainWindow_for_ImageAnalysis", "close"))
