#include "Header.h"

struct Tensor
{
	uint32_t D1, D2, D3, D4, D2Size, D4Size;
	float* forwardTensor, * backwardTensor;

	Tensor(uint32_t D1, uint32_t D2 = 1, uint32_t D3 = 1, uint32_t D4 = 1)
		: D1(D1), D2(D2), D3(D3), D4(D4)
	{
		this->D2Size = D1 * D2;
		this->D4Size = D2Size * D3 * D4;
		this->forwardTensor = new float[D4Size];
		this->backwardTensor = new float[D4Size];
	}

	~Tensor()
	{
		delete[] forwardTensor;
		delete[] backwardTensor;
	}
};

struct Loss
{
	float loss;
	virtual void Apply(Tensor* tensor) = 0;
	virtual void Print() = 0;
};

struct MSELoss : Loss
{
	void Apply(Tensor* tensor) override
	{
		assert(tensor != nullptr);
		assert(tensor->forwardTensor != nullptr);
		assert(tensor->backwardTensor != nullptr);
		assert(tensor->D4Size > 0);

		loss = 0.0f;
		float lossGradient;
		for (uint32_t i = 0; i < tensor->D4Size; i++)
		{
			lossGradient = 2 * (tensor->backwardTensor[i] - tensor->forwardTensor[i]);
			tensor->backwardTensor[i] = lossGradient;
			loss += abs(lossGradient);
		}
		loss /= tensor->D4Size * tensor->D1;
	}

	void Print() override
	{
		printf("MSELoss: %6.3f\n", loss);
	}
};

struct Operation
{
	virtual ~Operation() = default;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void Update(const float* learningRate) = 0;
	virtual void PrintForward() = 0;
	virtual void PrintBackward() = 0;
	virtual void PrintParameters() = 0;
	virtual Tensor* GetOutputTensor() = 0;
};

struct ReluOp : Operation
{
	Tensor* inputTensor, * outputTensor;

	ReluOp(Tensor* inputTensor)
		: inputTensor(inputTensor)
	{
		assert(inputTensor != nullptr);
		this->outputTensor = new Tensor(inputTensor->D1, inputTensor->D2, inputTensor->D3, inputTensor->D4);
	}

	~ReluOp()
	{
		delete outputTensor;
	}

	void Forward() override
	{
		cpuReluForward(inputTensor->D4Size, &ONEF, inputTensor->forwardTensor, &ZEROF, outputTensor->forwardTensor);
	}

	void Backward() override
	{
		cpuReluBackward(inputTensor->D4Size, &ONEF, outputTensor->forwardTensor, outputTensor->backwardTensor, inputTensor->forwardTensor, &ZEROF, inputTensor->backwardTensor);
	}

	void Update(const float* learningRate) override
	{
	}

	void PrintForward() override
	{
		PrintMatrixf32(outputTensor->forwardTensor, outputTensor->D1, outputTensor->D2, "Relu Op Forward");
	}

	void PrintBackward() override
	{
		PrintMatrixf32(outputTensor->backwardTensor, outputTensor->D1, outputTensor->D2, "Relu Op Backward");
	}

	void PrintParameters() override
	{
	}

	Tensor* GetOutputTensor() override
	{
		return outputTensor;
	}
};

struct LeakyReluOp : Operation
{
	Tensor* inputTensor, * outputTensor;

	LeakyReluOp(Tensor* inputTensor)
		: inputTensor(inputTensor)
	{
		assert(inputTensor != nullptr);
		this->outputTensor = new Tensor(inputTensor->D1, inputTensor->D2, inputTensor->D3, inputTensor->D4);
	}

	~LeakyReluOp()
	{
		delete outputTensor;
	}

	void Forward() override
	{
		cpuLeakyReluForward(inputTensor->D4Size, &ONEF, inputTensor->forwardTensor, &ZEROF, outputTensor->forwardTensor);
	}

	void Backward() override
	{
		cpuLeakyReluBackward(inputTensor->D4Size, &ONEF, outputTensor->forwardTensor, outputTensor->backwardTensor, inputTensor->forwardTensor, &ZEROF, inputTensor->backwardTensor);
	}

	void Update(const float* learningRate) override
	{
	}

	void PrintForward() override
	{
		PrintMatrixf32(outputTensor->forwardTensor, outputTensor->D1, outputTensor->D2, "Relu Op Forward");
	}

	void PrintBackward() override
	{
		PrintMatrixf32(outputTensor->backwardTensor, outputTensor->D1, outputTensor->D2, "Relu Op Backward");
	}

	void PrintParameters() override
	{
	}

	Tensor* GetOutputTensor() override
	{
		return outputTensor;
	}
};

struct WeightOp : Operation
{
	Tensor* inputTensor, * outputTensor, * weightTensor;

	WeightOp(Tensor* inputTensor, uint32_t D1 = 0)
		: inputTensor(inputTensor)
	{
		assert(inputTensor != nullptr);
		if (D1 <= 0)
		{
			D1 = inputTensor->D1;
			printf("Using input tensor D1 for weight tensor D1\n");
		}
		
		this->weightTensor = new Tensor(D1, inputTensor->D1);
		this->outputTensor = new Tensor(D1, inputTensor->D2, inputTensor->D3);
	
		for (uint32_t i = 0; i < weightTensor->D4Size; i++)
			weightTensor->forwardTensor[i] = RandomGaussian(0, sqrt(2.0f / D1));
	}

	~WeightOp()
	{
		delete weightTensor;
		delete outputTensor;
	}
	
	void Forward() override
	{
		cpuSgemmStridedBatched
		(
			false, false,
			outputTensor->D1, outputTensor->D2, inputTensor->D1,
			&ONEF,
			weightTensor->forwardTensor, weightTensor->D1, 0,
			inputTensor->forwardTensor, inputTensor->D1, inputTensor->D2Size,
			&ZEROF,
			outputTensor->forwardTensor, outputTensor->D1, outputTensor->D2Size,
			inputTensor->D3
		);
	}
	
	void Backward() override
	{
		memset(weightTensor->backwardTensor, 0, weightTensor->D4Size * sizeof(float));
		cpuSgemmStridedBatched
		(
			false, true,
			weightTensor->D1, weightTensor->D2, inputTensor->D2,
			&ONEF,
			outputTensor->backwardTensor, outputTensor->D1, outputTensor->D2Size,
			inputTensor->forwardTensor, inputTensor->D1, inputTensor->D2Size,
			&ONEF,
			weightTensor->backwardTensor, weightTensor->D1, 0,
			inputTensor->D3
		);
		
		cpuSgemmStridedBatched
		(
			true, false,
			inputTensor->D1, inputTensor->D2, outputTensor->D1,
			&ONEF,
			weightTensor->forwardTensor, weightTensor->D1, 0,
			outputTensor->backwardTensor, outputTensor->D1, outputTensor->D2Size,
			&ZEROF,
			inputTensor->backwardTensor, inputTensor->D1, inputTensor->D2Size,
			inputTensor->D3
		);
	}

	void Update(const float* learningRate) override
	{
		cpuSaxpy(weightTensor->D4Size, learningRate, weightTensor->backwardTensor, 1, weightTensor->forwardTensor, 1);
	}

	void PrintForward() override
	{
		PrintMatrixf32(outputTensor->forwardTensor, outputTensor->D1, outputTensor->D2, "Weight Op Forward");
	}

	void PrintBackward() override
	{
		PrintMatrixf32(outputTensor->backwardTensor, outputTensor->D1, outputTensor->D2, "Weight Op Backward");
	}

	void PrintParameters() override
	{
		PrintMatrixf32(weightTensor->forwardTensor, weightTensor->D1, weightTensor->D2, "Weight Parameters");
	}
	
	Tensor* GetOutputTensor() override
	{
		return outputTensor;
	}
};

struct NN
{
	std::vector<Tensor*> tensors;
	std::vector<Operation*> operations;

	~NN()
	{
		for (Tensor* tensor : tensors)
			delete tensor;
		for (Operation* operation : operations)
			delete operation;
	}

	Tensor* AddTensor(Tensor* tensor)
	{
		tensors.push_back(tensor);
		return tensor;
	}

	Tensor* AddOperation(Operation* operation)
	{
		operations.push_back(operation);
		return operation->GetOutputTensor();
	}

	void Forward()
	{
		for (Operation* operation : operations)
			operation->Forward();
	}

	void Backward()
	{
		for (auto it = operations.rbegin(); it != operations.rend(); it++)
			(*it)->Backward();
	}

	void Update(const float* learningRate)
	{
		for (Operation* operation : operations)
			operation->Update(learningRate);
	}

	void PrintForward()
	{
		for (Operation* operation : operations)
			operation->PrintForward();
	}

	void PrintBackward()
	{
		for (auto it = operations.rbegin(); it != operations.rend(); it++)
			(*it)->PrintBackward();
	}

	void PrintParameters()
	{
		for (Operation* operation : operations)
			operation->PrintParameters();
	}
};

int main()
{
	srand(time(NULL));

	const float LEARNING_RATE = 0.01f;
	const uint32_t BATCH_SIZE = 32;
	const uint32_t EPISODES = 1024;
	const uint32_t INPUT_SIZE = 2;

	float UPDATE_RATE = LEARNING_RATE * InvSqrt(BATCH_SIZE);

	MSELoss loss;
	NN nn;

	Tensor* input = nn.AddTensor(new Tensor(INPUT_SIZE, BATCH_SIZE));
	Tensor* product = nn.AddOperation(new WeightOp(input, 2 * INPUT_SIZE));
	Tensor* activation = nn.AddOperation(new LeakyReluOp(product));
	Tensor* output = nn.AddOperation(new WeightOp(activation, INPUT_SIZE));

	for (uint32_t episode = 0; episode < EPISODES; episode++)
	{
		for (uint32_t batch = 0; batch < BATCH_SIZE; batch++)
			for (uint32_t inputIndex = 0; inputIndex < INPUT_SIZE; inputIndex++)
				input->forwardTensor[batch * INPUT_SIZE + inputIndex] = RandomFloat();

		nn.Forward();

		for (uint32_t batch = 0; batch < BATCH_SIZE; batch++)
			for (uint32_t inputIndex = 0; inputIndex < INPUT_SIZE; inputIndex++)
				output->backwardTensor[batch * INPUT_SIZE + inputIndex] = input->forwardTensor[batch * INPUT_SIZE + inputIndex];

		loss.Apply(output);
		nn.Backward();

		nn.Update(&UPDATE_RATE);
		
		loss.Print();
	}
	printf("\n");

	/*PrintMatrixf32(input->forwardTensor, input->D1, input->D2, "Input Forward");
	nn.PrintForward();

	nn.PrintBackward();
	PrintMatrixf32(input->backwardTensor, input->D1, input->D2, "Input Backward");*/

	nn.PrintParameters();

	return 0;
}