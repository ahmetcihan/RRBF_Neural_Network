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
    //resize vectors
    numNeurons = numNeurons_;
    centers.resize(numNeurons);
    stdDevs.resize(numNeurons);
    weights.resize(numNeurons);

    // generate random values
    // centers : [-3, 3]
    // stdDevs: [0.1, 1.0]
    // weights: [-0.5, 0.5])
    // rng.generateDouble() generates value between 0 to 1

    QRandomGenerator rng(QTime::currentTime().msec());

    for (int i = 0; i < numNeurons; ++i) {
        centers[i] = rng.generateDouble() * 6.0 - 3.0;  // -3 to 3
        stdDevs[i] = rng.generateDouble() * 0.9 + 0.1;  // 0.1 to 1.0
        weights[i] = rng.generateDouble() - 0.5;        // -0.5 to 0.5
    }
    qDebug() << "check starting random values" << centers[numNeurons-1] << " , " << stdDevs[numNeurons-1] << " , " <<  weights[numNeurons-1];
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

void MainWindow::computeGradients(double x, double y, double y_desired, QVector<double>& grad_weights, QVector<double>& grad_stdDevs, QVector<double>& grad_centers) const
{
    grad_weights.resize(numNeurons);
    grad_stdDevs.resize(numNeurons);
    grad_centers.resize(numNeurons);
    grad_weights.fill(0.0);
    grad_stdDevs.fill(0.0);
    grad_centers.fill(0.0);

    double y_output = computeOutput(x, y);
    double error = y_desired - y_output;

    for (int i = 0; i < numNeurons; ++i) {
        //calculate phi_x and phi_y separately
        double phi_x = qExp(-(x - centers[i]) * (x - centers[i]) / (2 * stdDevs[i] * stdDevs[i]));
        double phi_y = qExp(-(y - centers[i]) * (y - centers[i]) / (2 * stdDevs[i] * stdDevs[i]));
        double phi_i = phi_x + phi_y;

        //gradient for weights (dE/d(w_i))
        grad_weights[i] = -error * phi_i;

        //gradient for standard deviations (dE/d(delta_i))
        double term1_std = (x - centers[i]) * (x - centers[i]) / (stdDevs[i] * stdDevs[i] * stdDevs[i]);
        double term2_std = (y - centers[i]) * (y - centers[i]) / (stdDevs[i] * stdDevs[i] * stdDevs[i]);
        grad_stdDevs[i] = -error * weights[i] * (phi_x * term1_std + phi_y * term2_std);

        //gradient for centers (dE/d(m_i))
        double term1_center = (x - centers[i]) / (stdDevs[i] * stdDevs[i]);
        double term2_center = (y - centers[i]) / (stdDevs[i] * stdDevs[i]);
        grad_centers[i] = -error * weights[i] * (phi_x * term1_center + phi_y * term2_center);
    }
}
void MainWindow::updateParameters(const QVector<double>& grad_weights, const QVector<double>& grad_stdDevs, const QVector<double>& grad_centers, double learningRate)
{
    for (int i = 0; i < numNeurons; ++i) {
        weights[i] -= learningRate * grad_weights[i];
        stdDevs[i] -= learningRate * grad_stdDevs[i];
        centers[i] -= learningRate * grad_centers[i];

        //standart deviation must stay positive
        if (stdDevs[i] < 0.01) stdDevs[i] = 0.01;
    }
}
void MainWindow::startTraining()
{
    if (training) return; //do not start traing if it is already runing
    training = true;

    initializeNetwork(ui->neuronSpinBox->value());

    //create training data for function f = (sin(x)/x)(sin(y)/y)
    trainingData.clear();
    for (double x = -3.0; x <= 3.0; x += 0.5) { //11 x 11 = 121 value
        for (double y = -3.0; y <= 3.0; y += 0.5) {

            double x_val = (x == 0.0) ? 0.0001 : x; //avoid divide by zero for x
            double y_val = (y == 0.0) ? 0.0001 : y; //avoid divide by zero for y

            double target = (qSin(x_val) / x_val) * (qSin(y_val) / y_val);

            trainingData.push_back({{x, y}, target});
        }
    }

    dataIndex = 0;

    //define 100 milisecond timer for training loop
    QTimer* timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(trainStep()));
    timer->start(100);
}
void MainWindow::stopTraining()
{
    training = false;

    qDebug() << "Training is finished. Learned Parameters:";
    qDebug() << "Neuron\tWeight\tCenter\tStdDev";

    for (int i = 0; i < weights.size(); ++i) {
        qDebug() << i + 1 << "\t" << weights[i] << "\t" << centers[i] << "\t" << stdDevs[i];
    }
}
void MainWindow::trainStep()
{
    if (training == false) {
        //stop timer
        if (QObject::sender()) {    //timer object calls this slot
            QTimer* timer = qobject_cast<QTimer*>(QObject::sender());
            if (timer) {
                timer->stop();
                delete timer;
            }
        }
        return;
    }

    double totalError = 0.0;
    double learningRate = ui->learningRateSpinBox->value();
    double stopCondition = ui->stopConditionSpinBox->value();
    QVector<double> grad_weights, grad_stdDevs, grad_centers;

    const auto& data = trainingData[dataIndex];
    double x = data.first.first;
    double y = data.first.second;
    double y_desired = data.second;

    double y_output = computeOutput(x, y);
    double error = y_desired - y_output;

    computeGradients(x, y, y_desired, grad_weights, grad_stdDevs, grad_centers);

    updateParameters(grad_weights, grad_stdDevs, grad_centers, learningRate);

    for (const auto& data : trainingData) {
        double x = data.first.first;
        double y = data.first.second;
        double y_desired = data.second;
        double y_output = computeOutput(x, y);
        double error = y_desired - y_output;
        totalError += 0.5 * error * error;
    }

    totalError /= trainingData.size();
    ui->errorLabel->setText(QString("Error: %1").arg(totalError, 0, 'f', 6));

    if (totalError < stopCondition) {
        training = false;
    }

    dataIndex = (dataIndex + 1) % static_cast<size_t>(trainingData.size());

    QApplication::processEvents();
}
MainWindow::~MainWindow()
{
    delete ui;

}

