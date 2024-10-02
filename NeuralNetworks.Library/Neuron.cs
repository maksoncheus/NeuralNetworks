////Нейроны обучаются по принципу обратного распространения ошибки (Back Propagation)
///Суть:
///1) Каждый датасет (набор данных, на котором обучается нейросеть) содержит все необходимые входные
///   сигналы, а также ОЖИДАЕМЫЙ результат.
///2) После "пробрасывания" всех сигналов нейросеть предсказывает ответ.
///3) Вычислить значение ошибки (если ожидаемый результат не совпал с фактическим)
///   ( error = actual - expected )
///4) Вычислить дельту (разницу) ( error * SigmoidDx(x) )
///5) Перерасчет весов !!!
///   Weights[i] = Weights[i] - Inputs[i] * delta * learningRate


namespace NeuralNetworks.Library
{
    /// <summary>
    /// Один отдельный нейрон
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Коллекция весов
        /// </summary>
        public List<double> Weights { get; }
        /// <summary>
        /// Коллекция входов
        /// </summary>
        public List<double> Inputs { get; }
        /// <summary>
        /// Тип нейрона
        /// </summary>
        public NeuronType NeuronType { get; }
        /// <summary>
        /// Вывод (значение, хранящееся в нейроне)
        /// </summary>
        public double Output { get; private set; }
        public double Delta { get; private set; }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputCount">Количество входов</param>
        /// <param name="type"><see cref="NeuronType">Тип нейрона</see>/></param>

        public Neuron(int inputCount, NeuronType type = NeuronType.Hidden)
        {
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeightsRandomValue(inputCount);
        }
        /// <summary>
        /// Генерация случайных весов
        /// </summary>
        /// <param name="inputCount">Количество входов</param>

        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }
        /// <summary>
        /// Метод для последовательной передачи сигналов от входа к выходу
        /// </summary>
        /// <param name="inputs">Коллекция входов</param>
        /// <returns></returns>
        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }
        /// <summary>
        /// Сигмоидная функция активации
        /// </summary>
        /// <param name="x">Сумма произведения весов и входов</param>
        /// <returns>Значение функции активации</returns>
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));
        /// <summary>
        /// Производная сигмоидной функции активации
        /// </summary>
        /// <param name="x"></param>
        /// <returns>Значение производной функции активации</returns>
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
                return;
            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] = Weights[i] - Inputs[i] * Delta * learningRate;
            }
        }
        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
