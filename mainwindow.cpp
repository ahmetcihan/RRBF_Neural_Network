#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setWindowTitle("MFA501 Assessment 2B - Ahmet Cihan AKINCA - RRBF Neural Network");

    connect(ui->startTrainingButton, SIGNAL(clicked(bool)), this, SLOT(startTraining()));
    connect(ui->stopTrainingButton, SIGNAL(clicked(bool)), this, SLOT(stopTraining()));
    connect(ui->createTrainingSetButton, SIGNAL(clicked(bool)), this, SLOT(createTrainingDataSet()));
    connect(ui->pushButton_find_Z, SIGNAL(clicked(bool)), this, SLOT(FindZ()));
    connect(ui->drawTestGraphButton, SIGNAL(clicked(bool)), this, SLOT(drawTestGraph())); // Yeni bağlantı

    training = false;
    epochCounter = 0;
    stepCounter = 0;
    errorHistory.clear();
    stepIndices.clear();

    customPlot = new QCustomPlot(this->ui->groupBox_training);
    customPlot->setGeometry(10, 90, 420, 365);
    customPlot->axisRect()->setAutoMargins(QCP::msNone);
    customPlot->axisRect()->setMargins(QMargins(70, 20, 20, 60));
    customPlot->xAxis->setLabel("Training Step");
    customPlot->xAxis->setLabelFont(QFont("Arial", 12));
    customPlot->xAxis->setLabelColor(Qt::black);
    customPlot->yAxis->setLabel("Mean Squared Error");
    customPlot->yAxis->setLabelFont(QFont("Arial", 12, QFont::Normal, false));
    customPlot->yAxis->setLabelColor(Qt::black);

    testPlot = new QCustomPlot(this->ui->groupBox_testing);
    testPlot->setGeometry(10, 180, 420, 450);
    testPlot->axisRect()->setAutoMargins(QCP::msNone);
    testPlot->axisRect()->setMargins(QMargins(70, 20, 20, 60));
    testPlot->xAxis->setLabel("Test Data Index");
    testPlot->xAxis->setLabelFont(QFont("Arial", 12));
    testPlot->xAxis->setLabelColor(Qt::black);
    testPlot->xAxis->setLabelPadding(10);
    testPlot->yAxis->setLabel("Z Value");
    testPlot->yAxis->setLabelFont(QFont("Arial", 12));
    testPlot->yAxis->setLabelColor(Qt::black);

    QFont tickFont("Arial", 8);
    tickFont.setWeight(QFont::Normal);
    tickFont.setItalic(false);

    customPlot->xAxis->setTickLabelFont(tickFont);
    customPlot->yAxis->setTickLabelFont(tickFont);
    testPlot->xAxis->setTickLabelFont(tickFont);
    testPlot->yAxis->setTickLabelFont(tickFont);
}

MainWindow::~MainWindow()
{
    delete ui;

}

