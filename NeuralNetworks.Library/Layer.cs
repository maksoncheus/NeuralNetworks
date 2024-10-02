namespace NeuralNetworks.Library
{
    /// <summary>
    /// Отдельный слой нейронов
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Коллекция нейронов на слое
        /// </summary>
        public List<Neuron> Neurons { get; }
        /// <summary>
        /// Количество нейронов на слое
        /// </summary>
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType Type;
        /// <summary>
        /// Слой нейронной сети
        /// </summary>
        /// <param name="neurons">Коллекция нейронов, которые будут помещены на слой</param>
        /// <param name="type">Тип слоя (все нейроны должны быть этого типа!)</param>
        /// <exception cref="ArgumentException">Один из нейронов оказался другого типа</exception>
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Hidden)
        {
            foreach (Neuron neuron in neurons)
            {
                if (neuron.NeuronType != type)
                    throw new ArgumentException("Некоторые нейроны не совпадают по типу со слоем"
                        ,nameof(neurons));
            }
            Neurons = neurons;
            Type = type;
        }
        /// <summary>
        /// Получить все сигналы на слое
        /// </summary>
        /// <returns>Коллекция сигналов на слое</returns>
        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
