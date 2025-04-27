#include "mainwindow.h"
#include "ui_mainwindow.h"

void MainWindow::FindZ()
{
    if (numNeurons == 0 || centers.isEmpty() || stdDevs.isEmpty() || weights.isEmpty()) {
        ui->zFoundLabel->setText("Z is found: N/A (Train the network first)");
        ui->zMustBeLabel->setText("Z must be: N/A (Train the network first)");
        return;
    }

    double x = ui->doubleSpinBox_test_x->value();
    double y = ui->doubleSpinBox_test_y->value();

    double z_found = computeOutput(x, y);

    double x_val = (x == 0.0) ? 0.0001 : x;
    double y_val = (y == 0.0) ? 0.0001 : y;
    double z_must_be = (qSin(x_val) / x_val) * (qSin(y_val) / y_val);

    ui->zFoundLabel->setText(QString("Z is found: %1").arg(z_found, 0, 'f', 6));
    ui->zMustBeLabel->setText(QString("Z must be: %1").arg(z_must_be, 0, 'f', 6));
}
void MainWindow::drawTestGraph()
{
    if (numNeurons == 0 || centers.isEmpty() || stdDevs.isEmpty() || weights.isEmpty()) {
        qDebug() << "Error: Network not trained yet!";
        return;
    }

    testIndices.clear();
    networkOutputs.clear();
    targetOutputs.clear();

    int index = 0;
    for (double x = -3.0; x <= 3.0; x += ui->doubleSpinBox_test_step_size->value()) { // Daha yoğun veri için 0.1 adımla
        for (double y = -3.0; y <= 3.0; y +=  ui->doubleSpinBox_test_step_size->value()) {
            double z_found = computeOutput(x, y);

            double x_val = (x == 0.0) ? 0.0001 : x;
            double y_val = (y == 0.0) ? 0.0001 : y;
            double z_must_be = (qSin(x_val) / x_val) * (qSin(y_val) / y_val);

            testIndices.append(index);
            networkOutputs.append(z_found);
            targetOutputs.append(z_must_be);
            index++;
        }
    }

    testPlot->clearGraphs();

    testPlot->addGraph();
    testPlot->graph(0)->setData(testIndices, networkOutputs);
    testPlot->graph(0)->setPen(QPen(Qt::red));
    testPlot->graph(0)->setLineStyle(QCPGraph::lsLine);
    testPlot->graph(0)->setName("Network Output");

    testPlot->addGraph();
    testPlot->graph(1)->setData(testIndices, targetOutputs);
    testPlot->graph(1)->setPen(QPen(Qt::blue));
    testPlot->graph(1)->setLineStyle(QCPGraph::lsLine);
    testPlot->graph(1)->setName("Target Output");

    testPlot->xAxis->setRange(0, testIndices.size());
    if (!networkOutputs.isEmpty() && !targetOutputs.isEmpty()) {
        double minNetwork = *std::min_element(networkOutputs.begin(), networkOutputs.end());
        double maxNetwork = *std::max_element(networkOutputs.begin(), networkOutputs.end());
        double minTarget = *std::min_element(targetOutputs.begin(), targetOutputs.end());
        double maxTarget = *std::max_element(targetOutputs.begin(), targetOutputs.end());
        double minY = qMin(minNetwork, minTarget);
        double maxY = qMax(maxNetwork, maxTarget);
        testPlot->yAxis->setRange(minY * 1.1, maxY * 1.1);
    } else {
        testPlot->yAxis->setRange(-1, 1);
    }

    testPlot->legend->setVisible(true);
    testPlot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignRight);

    testPlot->replot();
}
