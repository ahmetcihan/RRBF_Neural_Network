#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setWindowTitle("MFA501 Assessment 2B - Ahmet Cihan AKINCA");

    connect(ui->startTrainingButton, SIGNAL(clicked(bool)), this, SLOT(startTraining()));
    connect(ui->stopTrainingButton, SIGNAL(clicked(bool)), this, SLOT(stopTraining()));

    training = false;


}
void MainWindow::initializeNetwork(int numNeurons_)
{
    numNeurons = numNeurons_;
    centers.resize(numNeurons);
    stdDevs.resize(numNeurons);
    weights.resize(numNeurons);

    // Rastgele başlangıç değerleri (centers: [-3, 3], stdDevs: [0.1, 1.0], weights: [-0.5, 0.5])
    QRandomGenerator rng(QTime::currentTime().msec());
    for (int i = 0; i < numNeurons; ++i) {
        centers[i] = rng.generateDouble() * 6.0 - 3.0; // [-3, 3]
        stdDevs[i] = rng.generateDouble() * 0.9 + 0.1; // [0.1, 1.0]
        weights[i] = rng.generateDouble() - 0.5; // [-0.5, 0.5]
    }
}

double MainWindow::computePhi(int i, double x, double y) const
{
    double term1 = qExp(-(x - centers[i]) * (x - centers[i]) / (2 * stdDevs[i] * stdDevs[i]));
    double term2 = qExp(-(y - centers[i]) * (y - centers[i]) / (2 * stdDevs[i] * stdDevs[i]));
    return term1 + term2;
}

double MainWindow::computeOutput(double x, double y) const
{
    double output = 0.0;
    for (int i = 0; i < numNeurons; ++i) {
        output += weights[i] * computePhi(i, x, y);
    }
    return output;
}

void MainWindow::computeGradients(double x, double y, double y_desired,
                                 QVector<double>& grad_weights,
                                 QVector<double>& grad_stdDevs,
                                 QVector<double>& grad_centers) const
{
    grad_weights.resize(numNeurons);
    grad_weights.fill(0.0);
    grad_stdDevs.resize(numNeurons);
    grad_stdDevs.fill(0.0);
    grad_centers.resize(numNeurons);
    grad_centers.fill(0.0);

    // Compute output and error
    double y_output = computeOutput(x, y);
    double error = y_desired - y_output;

    for (int i = 0; i < numNeurons; ++i) {
        double phi_i = computePhi(i, x, y);

        // Gradient for weights (dE/d(w_i))
        grad_weights[i] = error * phi_i;

        // Gradient for standard deviations (dE/d(delta_i))
        double term1_std = (x - centers[i]) * (x - centers[i]) / (stdDevs[i] * stdDevs[i] * stdDevs[i]);
        double term2_std = (y - centers[i]) * (y - centers[i]) / (stdDevs[i] * stdDevs[i] * stdDevs[i]);
        grad_stdDevs[i] = error * weights[i] * phi_i * (term1_std + term2_std);

        // Gradient for centers (dE/d(m_i))
        double term1_center = (x - centers[i]) / (stdDevs[i] * stdDevs[i]);
        double term2_center = (y - centers[i]) / (stdDevs[i] * stdDevs[i]);
        grad_centers[i] = -error * weights[i] * phi_i * (term1_center + term2_center);
    }
}
void MainWindow::updateParameters(const QVector<double>& grad_weights,
                                 const QVector<double>& grad_stdDevs,
                                 const QVector<double>& grad_centers,
                                 double learningRate)
{
    for (int i = 0; i < numNeurons; ++i) {
        weights[i] -= learningRate * grad_weights[i];
        stdDevs[i] -= learningRate * grad_stdDevs[i];
        // Standart sapma pozitif kalmalı
        if (stdDevs[i] < 0.01) stdDevs[i] = 0.01;
        centers[i] -= learningRate * grad_centers[i];
    }
}
void MainWindow::startTraining()
{
    if (training) return; // Eğitim zaten devam ediyorsa tekrar başlatma
    training = true;
    ui->startTrainingButton->setEnabled(false);
    ui->stopTrainingButton->setEnabled(true);

    // Ağı başlat
    initializeNetwork(ui->neuronSpinBox->value());

    // Eğitim verisini hazırla: x, y in [-3, 3], 11x11 grid (121 veri noktası)
    trainingData.clear();
    for (double x = -3.0; x <= 3.0; x += 0.5) {
        for (double y = -3.0; y <= 3.0; y += 0.5) {
            double x_val = (x == 0.0) ? 0.0001 : x; // x=0'da tanımsızlıktan kaçın
            double y_val = (y == 0.0) ? 0.0001 : y; // y=0'da tanımsızlıktan kaçın
            double target = (qSin(x_val) / x_val) * (qSin(y_val) / y_val);
            trainingData.push_back({{x, y}, target});
        }
    }

    // dataIndex'i sıfırla
    dataIndex = 0;

    // Eğitim döngüsü için timer
    QTimer* timer = new QTimer(this);
    timer->setSingleShot(false); // Sürekli çalışsın
    connect(timer, &QTimer::timeout, this, &MainWindow::trainStep);
    timer->start(10); // Her 10ms'de bir trainStep çağrılacak
}
void MainWindow::stopTraining()
{
    training = false;
    ui->startTrainingButton->setEnabled(true);
    ui->stopTrainingButton->setEnabled(false);

    // Eğitim bittiğinde öğrenilen parametreleri yazdır
    qDebug() << "Training Stopped. Learned Parameters:";
    qDebug() << "Neuron\tWeight\tCenter\tStdDev";
    for (int i = 0; i < weights.size(); ++i) {
        qDebug() << i + 1 << "\t"
                 << weights[i] << "\t"
                 << centers[i] << "\t"
                 << stdDevs[i];
    }
}
void MainWindow::trainStep()
{
    if (!training) {
        // Timer'ı durdur ve parametreleri yazdır
        if (QObject::sender()) {
            QTimer* timer = qobject_cast<QTimer*>(QObject::sender());
            if (timer) {
                timer->stop();
                delete timer;
            }
        }
        // Parametreler stopTraining içinde yazdırıldığı için burada tekrar yazdırmaya gerek yok
        return;
    }

    double totalError = 0.0;
    double learningRate = ui->learningRateSpinBox->value();
    double stopCondition = ui->stopConditionSpinBox->value();
    QVector<double> grad_weights, grad_stdDevs, grad_centers;

    // Tek bir veri noktası için eğitim
    const auto& data = trainingData[dataIndex];
    double x = data.first.first;
    double y = data.first.second;
    double y_desired = data.second;

    // İleri yayılım
    double y_output = computeOutput(x, y);
    double error = y_desired - y_output;
    totalError += 0.5 * error * error;

    // Gradyanları hesapla
    computeGradients(x, y, y_desired, grad_weights, grad_stdDevs, grad_centers);

    // Parametreleri güncelle
    updateParameters(grad_weights, grad_stdDevs, grad_centers, learningRate);

    // Tüm veri noktaları için hata hesapla (her adımda güncel hata gösterimi için)
    totalError = 0.0;
    for (const auto& data : trainingData) {
        double x = data.first.first;
        double y = data.first.second;
        double y_desired = data.second;
        double y_output = computeOutput(x, y);
        double error = y_desired - y_output;
        totalError += 0.5 * error * error;
    }

    // Ortalama hata
    totalError /= trainingData.size();
    ui->errorLabel->setText(QString("Error: %1").arg(totalError, 0, 'f', 6));

    // Hata stop condition'ın altına düşerse eğitimi durdur
    if (totalError < stopCondition) {
        training = false;
    }

    // Bir sonraki veri noktasına geç
    dataIndex = (dataIndex + 1) % static_cast<size_t>(trainingData.size());

    // Arayüzün yanıt verebilir kalmasını sağlamak için olay döngüsüne izin ver
    QApplication::processEvents();
}
MainWindow::~MainWindow()
{
    delete ui;

}

