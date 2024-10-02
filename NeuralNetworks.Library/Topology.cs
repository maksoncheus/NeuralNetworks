using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.Library
{
    /// <summary>
    /// Топология сети
    /// </summary>
    public class Topology
    {
        /// <summary>
        /// Количество входных нейронов
        /// </summary>
        public int InputCount { get; }
        /// <summary>
        /// Количество выходных нейронов
        /// </summary>
        public int OutputCount { get; }
        /// <summary>
        /// Коллекция количества нейронов на скрытых слоях
        /// (каждый элемент коллекции - количество нейронов на каком-то из скрытых слоев)
        /// </summary>
        public List<int> HiddenLayers { get; }
        /// <summary>
        /// "Скорость" обучения. С каким шагом будет идти приращение функции.
        /// Чем больше - тем выше скорость обучения, но ниже точность, и наоборот
        /// </summary>
        public double LearningRate { get; }
        /// <summary>
        /// Топология сети
        /// </summary>
        /// <param name="inputCount">Количество входных нейронов</param>
        /// <param name="outputCount">Количество выходных нейронов</param>
        /// <param name="learningRate">Коллекция количества нейронов на скрытых слоях
        /// (каждый элемент коллекции - количество нейронов на каком-то из скрытых слоев)</param>
        /// <param name="hiddenLayers">"Скорость" обучения. С каким шагом будет идти приращение функции.
        /// Чем больше - тем выше скорость обучения, но ниже точность, и наоборот</param>
        public Topology(int inputCount, int outputCount, double learningRate, params int[]
            hiddenLayers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenLayers = new(hiddenLayers);
            LearningRate = learningRate;
        }
    }
}
