class Simulation {
    constructor() {
        this.env = new Environment('gameCanvas');
        // Make environment accessible to agents
        window.env = this.env;
        
        this.agent = null;
        this.chartUpdateTimeout = null;
        this.chart = this.initChart();
        this.running = false;
        this.episodeCount = 0;
        this.totalReward = 0;
        
        // Comprehensive metrics tracking for CSV export
        this.episodeMetrics = [];
        this.currentEpisodeSteps = 0;
        this.episodeStartTime = null;
        
        this.initEventListeners();
    }

    initChart() {
        const ctx = document.getElementById('rewardChart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Reward',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.1,
                    fill: false,
                    pointRadius: 2,
                    pointHoverRadius: 4
                }, {
                    label: 'Average Reward (last 10)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: false,
                    pointRadius: 1,
                    pointHoverRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            maxTicksLimit: 8
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        ticks: {
                            maxTicksLimit: 10,
                            callback: function(value) {
                                return Math.floor(value);
                            }
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                        },
                        pan: {
                            enabled: true,
                            mode: 'x',
                        }
                    }
                }
            }
        });
    }

    updateChart(reward) {
        try {
            // Add new reward data
            this.chart.data.labels.push(this.episodeCount);
            this.chart.data.datasets[0].data.push({x: this.episodeCount, y: reward});

            // Calculate and update moving average
            const recentRewards = this.chart.data.datasets[0].data.slice(-10);
            const avgReward = recentRewards.reduce((a, b) => a + (b.y || b), 0) / recentRewards.length;
            this.chart.data.datasets[1].data.push({x: this.episodeCount, y: avgReward});

            // Dynamic compression based on episode count
            const maxVisiblePoints = 100; // Maximum points to show at full detail
            
            if (this.chart.data.datasets[0].data.length > maxVisiblePoints) {
                // Compress older data by keeping every Nth point
                const compressionRatio = Math.ceil(this.chart.data.datasets[0].data.length / maxVisiblePoints);
                
                // Keep recent episodes at full resolution, compress older ones
                const recentPoints = 50; // Keep last 50 episodes at full resolution
                const fullResolutionData = this.chart.data.datasets[0].data.slice(-recentPoints);
                const fullResolutionAvg = this.chart.data.datasets[1].data.slice(-recentPoints);
                
                // Compress older data
                const olderData = this.chart.data.datasets[0].data.slice(0, -recentPoints);
                const olderAvgData = this.chart.data.datasets[1].data.slice(0, -recentPoints);
                
                const compressedData = [];
                const compressedAvgData = [];
                
                for (let i = 0; i < olderData.length; i += compressionRatio) {
                    if (olderData[i]) {
                        compressedData.push(olderData[i]);
                    }
                    if (olderAvgData[i]) {
                        compressedAvgData.push(olderAvgData[i]);
                    }
                }
                
                // Combine compressed older data with full resolution recent data
                this.chart.data.datasets[0].data = [...compressedData, ...fullResolutionData];
                this.chart.data.datasets[1].data = [...compressedAvgData, ...fullResolutionAvg];
                
                // Update labels to match
                this.chart.data.labels = this.chart.data.datasets[0].data.map(point => point.x);
            }

            // Update chart options for better scaling
            this.chart.options.scales.x.min = Math.max(1, Math.min(...this.chart.data.labels) - 5);
            this.chart.options.scales.x.max = Math.max(...this.chart.data.labels) + 5;
            
            // Set appropriate tick intervals based on data range
            const dataRange = this.chart.options.scales.x.max - this.chart.options.scales.x.min;
            this.chart.options.scales.x.ticks.stepSize = Math.max(1, Math.floor(dataRange / 10));

            this.chart.update('none'); // Update without animation for better performance
        } catch (error) {
            console.error('Error updating chart:', error);
        }
    }

    initEventListeners() {
        document.getElementById('startButton').addEventListener('click', () => {
            if (!this.running) {
                console.log('Starting training...');
            }
            this.toggleSimulation();
        });
        
        document.getElementById('resetButton').addEventListener('click', () => {
            console.log('Resetting simulation...');
            this.reset();
        });
        
        // Add CSV export button event listener
        document.getElementById('exportButton').addEventListener('click', () => {
            console.log('Exporting CSV report...');
            this.exportToCSV();
        });
        
        document.getElementById('algorithm').addEventListener('change', (e) => {
            console.log('Switching to algorithm:', e.target.value);
            if (this.agent) {
                this.reset();
            }
        });
        
        ['learning_rate', 'gamma', 'epsilon', 'speed'].forEach(param => {
            const element = document.getElementById(param);
            const valueDisplay = document.getElementById(`${param}_value`);
            element.addEventListener('input', () => {
                valueDisplay.textContent = element.value;
                if (this.agent) {
                    console.log('Updating parameter:', param, 'to', element.value);
                    this.agent.updateParams(this.getParams());
                }
            });
        });

        // Add max steps handler
        document.getElementById('max_steps').addEventListener('input', (e) => {
            this.env.setMaxSteps(parseInt(e.target.value));
        });

        // Handle algorithm change to show/hide appropriate parameters
        document.getElementById('algorithm').addEventListener('change', (e) => {
            const isActorCritic = e.target.value === 'actor_critic';
            document.getElementById('default_params').style.display = isActorCritic ? 'none' : 'block';
            document.getElementById('actor_critic_params').style.display = isActorCritic ? 'block' : 'none';
            if (this.agent) {
                this.reset();
            }
        });

        // Add actor-critic parameter handlers
        ['actor_learning_rate', 'critic_learning_rate', 'ac_gamma', 'temperature'].forEach(param => {
            const element = document.getElementById(param);
            const valueDisplay = document.getElementById(`${param}_value`);
            element.addEventListener('input', () => {
                valueDisplay.textContent = element.value;
                if (this.agent) {
                    this.agent.updateParams(this.getParams());
                }
            });
        });
    }

    getParams() {
        const algorithm = document.getElementById('algorithm').value;
        
        if (algorithm === 'actor_critic') {
            return {
                actorLearningRate: parseFloat(document.getElementById('actor_learning_rate').value),
                criticLearningRate: parseFloat(document.getElementById('critic_learning_rate').value),
                gamma: parseFloat(document.getElementById('ac_gamma').value),
                temperature: parseFloat(document.getElementById('temperature').value)
            };
        }

        return {
            learningRate: parseFloat(document.getElementById('learning_rate').value),
            gamma: parseFloat(document.getElementById('gamma').value),
            epsilon: parseFloat(document.getElementById('epsilon').value)
        };
    }

    createAgent() {
        try {
            const algorithm = document.getElementById('algorithm').value;
            const params = this.getParams();
            console.log('Creating agent:', algorithm, params);

            let agent;
            switch(algorithm) {
                case 'monte_carlo':
                    agent = new MonteCarloAgent(params);
                    break;
                case 'q_learning':
                    agent = new QLearningAgent(params);
                    break;
                case 'actor_critic':
                    agent = new ActorCriticAgent(params);
                    break;
                default:
                    console.warn('Unknown algorithm:', algorithm, 'defaulting to Q-Learning');
                    agent = new QLearningAgent(params);
            }

            // Initialize agent
            agent.reset();
            
            // Update environment with new algorithm
            this.env.setAlgorithm(algorithm);
            
            console.log('Agent created successfully:', algorithm);
            return agent;
        } catch (error) {
            console.error('Error creating agent:', error);
            // Default to Q-Learning if agent creation fails
            return new QLearningAgent(this.getParams());
        }
    }

    toggleSimulation() {
        try {
            if (this.running) {
                console.log('Pausing simulation');
                this.running = false;
                document.getElementById('startButton').textContent = 'Start Training';
                this.env.enableEditing();
            } else {
                console.log('Starting simulation');
                
                // Validate grid setup first
                if (!this.env.checkInitialization()) {
                    console.log('Grid initialization failed, remaining in edit mode');
                    return;
                }
                
                // Create fresh agent and start training
                this.agent = this.createAgent();
                if (!this.agent) {
                    alert('Failed to initialize agent. Please try a different algorithm.');
                    return;
                }

                // Initialize status panel
                document.getElementById('episodeCounter').textContent = '0';
                document.getElementById('totalEpisodes').textContent = document.getElementById('episodes').value;
                document.getElementById('currentReward').textContent = '0';
                document.getElementById('bestReward').textContent = '0';
                document.getElementById('averageReward').textContent = '0';
                document.getElementById('trashCount').textContent = this.env.trashCount;
                document.getElementById('successfulCompletions').textContent = this.env.successfulCompletions;

                this.running = true;
                document.getElementById('startButton').textContent = 'Pause Training';
                this.env.disableEditing();
                requestAnimationFrame(() => this.runEpisode());
            }
        } catch (error) {
            console.error('Error in toggleSimulation:', error);
            this.running = false;
            document.getElementById('startButton').textContent = 'Start Training';
            this.env.enableEditing();
            alert('An error occurred while starting the training. Please check the console for details.');
        }
    }

    reset() {
        this.running = false;
        document.getElementById('startButton').textContent = 'Start Training';
        this.agent = this.createAgent();
        this.episodeCount = 0;
        this.totalReward = 0;
        
        // Reset all metrics
        this.episodeMetrics = [];
        this.env.successfulCompletions = 0;
        
        this.env.resetHeatmap();
        this.env.reset();
        this.env.enableEditing(); // Enable editing after reset
        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.data.datasets[1].data = [];
        this.chart.update();
        
        // Update status panel to reflect reset
        this.updateStatusPanel();
    }

    updateStatusPanel() {
        try {
            document.getElementById('episodeCounter').textContent = this.episodeCount;
            document.getElementById('totalEpisodes').textContent = document.getElementById('episodes').value;
            document.getElementById('currentReward').textContent = this.totalReward.toFixed(1);
            document.getElementById('trashCount').textContent = this.env.trashCount;
            
            // Update successful completions counter
            document.getElementById('successfulCompletions').textContent = this.env.successfulCompletions;

            // Update best and average rewards - fix NaN issue
            const allRewards = this.chart.data.datasets[0].data;
            if (allRewards.length > 0) {
                // Extract reward values from chart data (handle both number and {x,y} object formats)
                const rewardValues = allRewards.map(item => 
                    typeof item === 'object' ? item.y : item
                ).filter(val => !isNaN(val) && val !== undefined);
                
                if (rewardValues.length > 0) {
                    const bestReward = Math.max(...rewardValues);
                    const recentRewards = rewardValues.slice(-10);
                    const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
                    
                    document.getElementById('bestReward').textContent = bestReward.toFixed(1);
                    document.getElementById('averageReward').textContent = avgReward.toFixed(1);
                } else {
                    document.getElementById('bestReward').textContent = '0.0';
                    document.getElementById('averageReward').textContent = '0.0';
                }
            } else {
                document.getElementById('bestReward').textContent = '0.0';
                document.getElementById('averageReward').textContent = '0.0';
            }

            // Trigger a render to update heatmap
            this.env.render();
        } catch (error) {
            console.error('Error updating status panel:', error);
        }
    }

    // CSV Export functionality
    exportToCSV() {
        if (this.episodeMetrics.length === 0) {
            alert('No training data available to export. Please run some episodes first.');
            return;
        }

        const headers = [
            'Episode',
            'Algorithm',
            'Current Reward',
            'Average Reward (Last 10)',
            'Steps to Complete',
            'Episode Time (s)',
            'Trash Remaining',
            'Initial Trash Count',
            'Trash Collected',
            'Is Successful Completion',
            'Total Successful Completions',
            'Learning Rate',
            'Epsilon',
            'Gamma',
            'Evil Robot Enabled',
            'Max Steps',
            'Timestamp'
        ];

        const csvContent = [
            headers.join(','),
            ...this.episodeMetrics.map(metric => [
                metric.episode,
                metric.algorithm,
                metric.currentReward.toFixed(2),
                metric.averageReward.toFixed(2),
                metric.stepsToComplete,
                metric.episodeTime.toFixed(2),
                metric.trashRemaining,
                metric.initialTrashCount,
                metric.trashCollected,
                metric.isSuccessfulCompletion,
                metric.totalSuccessfulCompletions,
                metric.learningRate,
                metric.epsilon,
                metric.gamma,
                metric.evilRobotEnabled,
                metric.maxSteps,
                metric.timestamp
            ].join(','))
        ].join('\n');

        // Create and download the CSV file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        
        // Generate filename with timestamp and algorithm
        const algorithm = document.getElementById('algorithm').value;
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
        link.setAttribute('download', `RL_Training_Report_${algorithm}_${timestamp}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`Exported ${this.episodeMetrics.length} episodes to CSV`);
        alert(`Successfully exported ${this.episodeMetrics.length} episodes to CSV!`);
    }

    async runEpisode() {
        try {
            if (!this.running) return;
            if (!this.agent) {
                console.error('No agent initialized!');
                this.running = false;
                document.getElementById('startButton').textContent = 'Start Training';
                return;
            }

            this.episodeCount++;
            let state = this.env.reset();
            this.totalReward = 0;
            let done = false;
            
            // Track episode metrics
            this.currentEpisodeSteps = 0;
            this.episodeStartTime = Date.now();
            const initialTrashCount = this.env.trashCount;
            let previousSuccessCount = this.env.successfulCompletions;

            // Update initial status
            this.updateStatusPanel();

            while (!done && this.running) {
                try {
                    // Get and execute action
                    const action = this.agent.selectAction(state);
                    const [nextState, reward, isDone] = this.env.step(action);
                    
                    // Update agent
                    const algorithm = document.getElementById('algorithm').value;
                    if (algorithm === 'monte_carlo') {
                        this.agent.update(state, action, reward);
                    } else if (algorithm === 'q_learning') {
                        this.agent.update(state, action, reward, nextState);
                    } else {  // actor_critic
                        this.agent.update(state, action, reward, nextState, isDone);
                    }

                    this.totalReward += reward;
                    this.currentEpisodeSteps++;
                    state = nextState;
                    done = isDone;

                    // Update status and visualization
                    this.updateStatusPanel();

                    // Control speed
                    const speed = parseInt(document.getElementById('speed').value);
                    if (speed < 60) {
                        await new Promise(r => setTimeout(r, 1000 / speed));
                    } else {
                        await new Promise(r => requestAnimationFrame(r));
                    }
                } catch (stepError) {
                    console.error('Error during episode step:', stepError);
                    done = true;
                }
            }

            // Episode cleanup
            if (document.getElementById('algorithm').value === 'monte_carlo') {
                this.agent.episodeEnd();
            }

            // Record episode metrics
            const episodeTime = (Date.now() - this.episodeStartTime) / 1000;
            const isSuccessful = this.env.successfulCompletions > previousSuccessCount;
            const allRewards = this.chart.data.datasets[0].data.concat([this.totalReward]);
            const avgReward = allRewards.slice(-10).reduce((a, b) => a + b, 0) / Math.min(allRewards.length, 10);
            
            const episodeMetric = {
                episode: this.episodeCount,
                algorithm: document.getElementById('algorithm').value,
                currentReward: this.totalReward,
                averageReward: avgReward,
                stepsToComplete: this.currentEpisodeSteps,
                episodeTime: episodeTime,
                trashRemaining: this.env.trashCount,
                initialTrashCount: initialTrashCount,
                trashCollected: initialTrashCount - this.env.trashCount,
                isSuccessfulCompletion: isSuccessful,
                totalSuccessfulCompletions: this.env.successfulCompletions,
                learningRate: this.getParams().learningRate || this.getParams().actorLearningRate || 'N/A',
                epsilon: this.getParams().epsilon || 'N/A',
                gamma: this.getParams().gamma,
                evilRobotEnabled: document.getElementById('evilRobotEnabled').checked,
                maxSteps: this.env.maxSteps,
                timestamp: new Date().toISOString()
            };
            
            this.episodeMetrics.push(episodeMetric);

            // Update chart with a slight delay to prevent performance issues
            if (this.chartUpdateTimeout) {
                clearTimeout(this.chartUpdateTimeout);
            }
            this.chartUpdateTimeout = setTimeout(() => {
                this.updateChart(this.totalReward);
            }, 16);

            // Continue to next episode or stop
            if (this.running && this.episodeCount < parseInt(document.getElementById('episodes').value)) {
                requestAnimationFrame(() => this.runEpisode());
            } else {
                this.running = false;
                document.getElementById('startButton').textContent = 'Start Training';
                this.env.enableEditing();
            }
        } catch (error) {
            console.error('Fatal error in runEpisode:', error);
            this.running = false;
            document.getElementById('startButton').textContent = 'Start Training';
            this.env.enableEditing();
        }
    }
}

// Start simulation when page loads
window.addEventListener('load', () => {
    const simulation = new Simulation();
});