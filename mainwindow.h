#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QApplication>
#include <QTimer>
#include <QTime>
#include <QDebug>
#include <QString>
#include <QtMath>
#include <QRandomGenerator>
#include "qcustomplot.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    // RRBF ağ parametreleri
    QVector<double> centers; // m_i
    QVector<double> stdDevs; // delta_i
    QVector<double> weights; // w_i
    int numNeurons;
    bool training;
    size_t dataIndex; // Eğitim döngüsünde hangi veri noktasının işlendiğini takip eder
    int epochCounter;       // Yeni epoch sayacı
    QCustomPlot* customPlot;
    QVector<double> errorHistory; // Hata değerlerini saklamak için
    QVector<double> stepIndices;  // Adım indekslerini saklamak için
    int stepCounter;              // Adım sayacı
    // Eğitim verisi
    QVector<QPair<QPair<double, double>, double>> trainingData;

    // RRBF işlevleri
    void initializeNetwork(int numNeurons_);
    double computePhi(int i, double x, double y) const;
    double computeOutput(double x, double y) const;
    void computeGradients(double x, double y, double y_desired,
                      QVector<double>& grad_weights,
                      QVector<double>& grad_stdDevs,
                      QVector<double>& grad_centers) const;
    void updateParameters(const QVector<double>& grad_weights,
                       const QVector<double>& grad_stdDevs,
                       const QVector<double>& grad_centers,
                       double learningRate);

private slots:
    void createTrainingDataSet();
    void startTraining();
    void stopTraining();
    void trainStep();
    void drawGraph();
    void FindZ();

};
#endif // MAINWINDOW_H
