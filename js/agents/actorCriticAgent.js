class ActorCriticAgent extends BaseAgent {
    constructor(params) {
        super(params);
        console.log('Initializing Actor-Critic agent with params:', params);
        
        // Store separate learning rates for actor and critic
        this.actorLearningRate = params.actorLearningRate || 0.001;
        this.criticLearningRate = params.criticLearningRate || 0.01;
        this.temperature = params.temperature || 1.0;  // Temperature for softmax
        
        // Initialize networks
        this.actorNetwork = new NeuralNetwork([24, 32, 4]);
        this.criticNetwork = new NeuralNetwork([24, 32, 1]);
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.episodeCount = 0;
    }

    // Convert state to neural network input
    stateToTensor(state) {
        try {
            if (!state || !state.wallE || !state.evilRobot) {
                console.error('Invalid state object:', state);
                return new Array(24).fill(0);
            }
            
            const input = new Array(24).fill(0);
            // Wall-E position (one-hot)
            const wallEIndex = state.wallE.y * 6 + state.wallE.x;
            input[wallEIndex] = 1;
            // Evil robot position (one-hot)
            const robotIndex = 12 + state.evilRobot.y * 6 + state.evilRobot.x;
            input[robotIndex] = 1;
            
            return input;
        } catch (error) {
            console.error('Error in stateToTensor:', error);
            return new Array(24).fill(0);
        }
    }

    selectAction(state) {
        try {
            const input = this.stateToTensor(state);
            const actionProbs = this.actorNetwork.forward(input);
            
            // Update state value in environment using critic's estimate
            const stateValue = this.criticNetwork.forward(input)[0];
            window.env.updateStateValue(state.wallE.x, state.wallE.y, stateValue);
            
            // Apply softmax with temperature
            const expProbs = actionProbs.map(x => Math.exp(x / this.temperature));
            const sum = expProbs.reduce((a, b) => a + b, 0);
            const probabilities = expProbs.map(x => x / sum);

            // Epsilon-greedy exploration
            if (Math.random() < this.epsilon) {
                const action = Math.floor(Math.random() * this.actionSpace);
                console.log('Random action (epsilon-greedy):', action);
                return action;
            }

            // Sample from policy
            const rand = Math.random();
            let cumSum = 0;
            for (let i = 0; i < probabilities.length; i++) {
                cumSum += probabilities[i];
                if (rand < cumSum) {
                    console.log('Policy action:', i, 'probabilities:', probabilities);
                    return i;
                }
            }
            console.log('Defaulting to last action');
            return probabilities.length - 1;
        } catch (error) {
            console.error('Error in selectAction:', error);
            return Math.floor(Math.random() * this.actionSpace);
        }
    }

    update(state, action, reward, nextState, done) {
        try {
            // Store transition
            this.states.push(this.stateToTensor(state));
            this.actions.push(action);
            this.rewards.push(reward);

            if (done) {
                this.episodeEnd();
            }
        } catch (error) {
            console.error('Error in update:', error);
        }
    }

    episodeEnd() {
        try {
            // Calculate returns and advantages
            const values = this.states.map(state => this.criticNetwork.forward(state)[0]);
            const returns = [];
            let G = 0;
            
            // Calculate returns in reverse
            for (let t = this.rewards.length - 1; t >= 0; t--) {
                G = this.rewards[t] + this.gamma * G;
                returns.unshift(G);
            }

            // Calculate advantages
            const advantages = returns.map((G, i) => G - values[i]);

            // Update networks
            for (let t = 0; t < this.states.length; t++) {
                // Update critic using critic learning rate
                this.criticNetwork.backward(
                    this.states[t], 
                    [returns[t]], 
                    this.criticLearningRate
                );

                // Update actor using actor learning rate and temperature-scaled advantages
                const actionProbs = this.actorNetwork.forward(this.states[t]);
                const actionGradients = actionProbs.map((_, i) => 
                    i === this.actions[t] ? advantages[t] / this.temperature : 0
                );
                this.actorNetwork.backward(
                    this.states[t],
                    actionGradients,
                    this.actorLearningRate
                );
            }

            // Log episode statistics
            this.episodeCount++;
            const totalReward = this.rewards.reduce((a, b) => a + b, 0);
            console.log('Episode', this.episodeCount, 'ended. Total reward:', totalReward);
            this.addReward(totalReward);

            // Clear episode data
            this.states = [];
            this.actions = [];
            this.rewards = [];
        } catch (error) {
            console.error('Error in episodeEnd:', error);
            // Reset episode data on error
            this.states = [];
            this.actions = [];
            this.rewards = [];
        }
    }

    updateParams(params) {
        // Update actor-critic specific parameters
        if (params.actorLearningRate !== undefined) {
            this.actorLearningRate = params.actorLearningRate;
        }
        if (params.criticLearningRate !== undefined) {
            this.criticLearningRate = params.criticLearningRate;
        }
        if (params.gamma !== undefined) {
            this.gamma = params.gamma;
        }
        if (params.temperature !== undefined) {
            this.temperature = params.temperature;
        }
    }

    reset() {
        try {
            super.reset();
            console.log('Resetting Actor-Critic agent');
            this.actorNetwork = new NeuralNetwork([24, 32, 4]);
            this.criticNetwork = new NeuralNetwork([24, 32, 1]);
            this.states = [];
            this.actions = [];
            this.rewards = [];
            this.episodeCount = 0;
        } catch (error) {
            console.error('Error in reset:', error);
        }
    }
}

// Simple neural network implementation
class NeuralNetwork {
    constructor(layers) {
        this.weights = [];
        this.biases = [];
        
        // Initialize weights and biases
        for (let i = 0; i < layers.length - 1; i++) {
            this.weights.push(
                Array(layers[i]).fill().map(() => 
                    Array(layers[i + 1]).fill().map(() => 
                        Math.random() * 2 - 1
                    )
                )
            );
            this.biases.push(Array(layers[i + 1]).fill(0));
        }
    }

    forward(input) {
        let current = input;
        
        // Forward propagation through layers
        for (let i = 0; i < this.weights.length; i++) {
            const layer = [];
            for (let j = 0; j < this.weights[i][0].length; j++) {
                let sum = this.biases[i][j];
                for (let k = 0; k < this.weights[i].length; k++) {
                    sum += current[k] * this.weights[i][k][j];
                }
                layer.push(i === this.weights.length - 1 ? sum : Math.max(0, sum)); // ReLU except last layer
            }
            current = layer;
        }
        
        return current;
    }

    backward(input, gradients, learningRate) {
        // Simple SGD update
        const output = this.forward(input);
        
        for (let i = this.weights.length - 1; i >= 0; i--) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    const grad = gradients[k] * input[j];
                    this.weights[i][j][k] += learningRate * grad;
                }
            }
            
            for (let j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] += learningRate * gradients[j];
            }
        }
    }
}