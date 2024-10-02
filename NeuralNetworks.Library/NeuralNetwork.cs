namespace NeuralNetworks.Library
{
    /// <summary>
    /// Нейронная сеть
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// Топология сети
        /// </summary>
        public Topology Topology { get; }
        /// <summary>
        /// Слои нейронной сети
        /// </summary>
        public List<Layer> Layers { get; }
        /// <summary>
        /// Нейронная сеть
        /// </summary>
        /// <param name="topology">Топология сети</param>
        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }
        public Neuron Predict(params double[] inputSignals)
        {
            SendSignalToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Массив входов</param>
        /// <param name="expected">Массив ожидаемых результатов (построчно)</param>
        /// <param name="epoch">Одна эпоха - когда весь датасет пройден</param>
        /// <returns>Средняя величина ошибки</returns>
        public double Learn(double[,] inputs, double[] expected, int epoch)
        {
            double error = 0;
            for(int i = 0; i < epoch; i++)
            {
                for(int j = 0; j < expected.Length; j++)
                {
                    double[] input = GetRow(inputs, j);

                    error += BackPropagation(expected[j], input);
                }
            }
            return error / epoch;
        }

        public static double[] GetRow(double[,] matrix, int j)
        {
            int columns = matrix.GetLength(1);
            double[] array = new double[columns];
            for (int k = 0; k < columns; k++)
                array[k] = matrix[j, k];
            return array;
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            double actualResult = Predict(inputs).Output;

            double difference = actualResult - expected;
            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }
            for(int i = Layers.Count - 2; i >= 0; i--)
            {
                Layer layer = Layers[i];
                Layer previousLayer = Layers[i + 1];
                for (int j = 0; j < layer.NeuronCount; j++)
                {
                    Neuron neuron = layer.Neurons[j];
                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        Neuron previousNeuron = previousLayer.Neurons[k];
                        double error = previousNeuron.Weights[j] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            return difference * difference;
        }
        /// <summary>
        /// Вызвать метод "пробрасывания" на всех слоях кроме входного
        /// </summary>
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                List<double> previousLayerSignals = Layers[i-1].GetSignals();
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }
        /// <summary>
        /// Послать входные сигналы
        /// </summary>
        /// <param name="signals">Список сигналов</param>
        private void SendSignalToInputNeurons(params double[] signals)
        {
            for (int i = 0; i < signals.Length; i++)
            {
                List<double> signal = new() { signals[i] };
                Neuron neuron = Layers[0].Neurons[i];
                neuron.FeedForward(signal);
            }
        }
        /// <summary>
        /// Создать входной слой
        /// </summary>
        private void CreateInputLayer()
        {
            List<Neuron> inputNeurons = new();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                Neuron neuron = new(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            Layer inputLayer = new(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
        /// <summary>
        /// Создать все скрытые слои
        /// </summary>
        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                List<Neuron> hiddenNeurons = new();
                Layer lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    Neuron neuron = new(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                Layer hiddenLayer = new(hiddenNeurons, NeuronType.Hidden);
                Layers.Add(hiddenLayer);
            }
        }
        /// <summary>
        /// Создать выходной слой
        /// </summary>
        private void CreateOutputLayer()
        {
            List<Neuron> outputNeurons = new();
            Layer lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                Neuron neuron = new(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            Layer outputLayer = new(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }
    }
}
