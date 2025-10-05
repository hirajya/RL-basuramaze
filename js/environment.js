class Environment {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Grid configuration - increased cell size from 80 to 100
        this.gridSize = 6;
        this.cellSize = 100;
        this.canvas.width = this.gridSize * this.cellSize;
        this.canvas.height = this.gridSize * this.cellSize;

        // Game elements
        this.EMPTY = 0;
        this.WALL = 1;
        this.TRASH = 2;
        this.MINE = 3;
        this.EXIT = 4;

        // Initialize grid
        this.reset();

        // Load assets
        this.loadAssets();

        // Add heatmap tracking
        this.stateVisits = Array(this.gridSize).fill().map(() => 
            Array(this.gridSize).fill(0)
        );
        this.maxVisits = 1;

        // Store custom grid configuration
        this.customGrid = null;
        this.customWallE = null;
        this.customEvilRobot = null;
        
        // Add max steps
        this.maxSteps = 200;  // Default value
        this.currentSteps = 0;
        
        // Add click handling for grid editing
        this.setupGridEditor();

        // Add state values and visited states tracking
        this.stateValues = Array(this.gridSize).fill().map(() => 
            Array(this.gridSize).fill(0)
        );
        this.visitedStates = Array(this.gridSize).fill().map(() => 
            Array(this.gridSize).fill(false)
        );
        this.minStateValue = -20;  // Initialize with reasonable defaults
        this.maxStateValue = 50;   // Based on reward structure

        // Add evil robot movement delay
        this.evilRobotMoveCounter = 0;
        this.evilRobotMoveDelay = 3;  // Move every 3 steps
    }

    checkInitialization() {
        try {
            // Check for required agents
            if (!this.wallE || !this.evilRobot) {
                console.error('Missing Wall-E or Evil Robot');
                alert('Please place both Wall-E and Evil Robot on the grid');
                this.enableEditing();
                return false;
            }

            // Check for exit point
            let hasExit = false;
            let hasTrash = false;
            for (let y = 0; y < this.gridSize; y++) {
                for (let x = 0; x < this.gridSize; x++) {
                    if (this.grid[y][x] === this.EXIT) {
                        hasExit = true;
                    }
                    if (this.grid[y][x] === this.TRASH) {
                        hasTrash = true;
                    }
                }
            }

            if (!hasExit) {
                console.error('Missing exit point');
                alert('Please place an exit point on the grid');
                this.enableEditing();
                return false;
            }

            if (!hasTrash) {
                console.error('No trash placed');
                alert('Please place at least one piece of trash on the grid');
                this.enableEditing();
                return false;
            }

            // Update trash count
            this.updateTrashCount();
            return true;
        } catch (error) {
            console.error('Error in checkInitialization:', error);
            return false;
        }
    }

    loadAssets() {
        this.assets = {};
        const assetNames = ['wall-e', 'evil-robot', 'trash', 'exit', 'mine', 'wall'];
        let loadedCount = 0;
        
        assetNames.forEach(name => {
            const img = new Image();
            img.onload = () => {
                loadedCount++;
                if (loadedCount === assetNames.length) {
                    this.render(); // Render once all images are loaded
                }
            };
            img.src = `assets/${name}.png`;
            this.assets[name] = img;
        });
    }

    reset() {
        // Initialize empty grid
        this.grid = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(this.EMPTY));
        
        if (this.customGrid) {
            // Use custom layout if available
            this.grid = this.customGrid.map(row => [...row]);
            this.wallE = {...this.customWallE};
            this.evilRobot = {...this.customEvilRobot};
            this.updateTrashCount();
        } else {
            // Use default layout
            // Place walls
            this.grid[1][1] = this.WALL;
            this.grid[2][3] = this.WALL;
            this.grid[4][4] = this.WALL;

            // Place trash
            this.grid[0][2] = this.TRASH;
            this.grid[3][1] = this.TRASH;
            this.grid[5][3] = this.TRASH;
            this.trashCount = 3;

            // Place mines
            this.grid[2][2] = this.MINE;
            this.grid[4][1] = this.MINE;

            // Place exit
            this.grid[5][5] = this.EXIT;

            // Initialize positions
            this.wallE = { x: 0, y: 0 };
            this.evilRobot = { x: Math.floor(this.gridSize/2), y: Math.floor(this.gridSize/2) };
        }
        
        this.evilRobotDirection = 0; // 0: up, 1: right, 2: down, 3: left
        this.evilRobotMoveCounter = 0;  // Reset movement counter
        this.currentSteps = 0;
        return this.getState();
    }

    getState() {
        return {
            grid: this.grid.map(row => [...row]),
            wallE: {...this.wallE},
            evilRobot: {...this.evilRobot},
            trashCount: this.trashCount
        };
    }

    step(action) {
        try {
            this.currentSteps++;
            if (this.currentSteps >= this.maxSteps) {
                return [this.getState(), -1, true];  // End episode if max steps reached
            }

            // Validate action
            if (typeof action !== 'number' || action < 0 || action > 3) {
                console.error('Invalid action:', action);
                action = Math.floor(Math.random() * 4); // Use random action as fallback
            }

            // Store previous position
            const prevPos = {...this.wallE};
            
            // Move Wall-E (0: up, 1: right, 2: down, 3: left)
            const moves = [[-1, 0], [0, 1], [1, 0], [0, -1]];
            const [dy, dx] = moves[action];
            const newY = this.wallE.y + dy;
            const newX = this.wallE.x + dx;

            // Check if move is valid
            if (this.isValidMove(newY, newX)) {
                this.wallE.y = newY;
                this.wallE.x = newX;
            }

            // Mark state as visited when agent moves there
            this.visitedStates[this.wallE.y][this.wallE.x] = true;

            // Calculate reward and check terminal state
            const reward = this.calculateReward(prevPos);
            const done = this.isTerminal();

            // Move evil robot if game not done and it's time to move
            if (!done) {
                this.evilRobotMoveCounter++;
                if (this.evilRobotMoveCounter >= this.evilRobotMoveDelay) {
                    this.moveEvilRobot();
                    this.evilRobotMoveCounter = 0;
                }
            }

            // Check collision with evil robot after its move
            if (this.wallE.x === this.evilRobot.x && this.wallE.y === this.evilRobot.y) {
                return [this.getState(), -50, true];
            }

            // Update heatmap
            this.stateVisits[this.wallE.y][this.wallE.x]++;
            this.maxVisits = Math.max(this.maxVisits, this.stateVisits[this.wallE.y][this.wallE.x]);

            return [this.getState(), reward, done];
        } catch (error) {
            console.error('Error in environment step:', error);
            // Return safe default values
            return [this.getState(), -1, true];
        }
    }

    isValidMove(y, x) {
        return (
            x >= 0 && x < this.gridSize &&
            y >= 0 && y < this.gridSize &&
            this.grid[y][x] !== this.WALL
        );
    }

    calculateReward(prevPos) {
        const currentCell = this.grid[this.wallE.y][this.wallE.x];
        
        // Hit wall (didn't move)
        if (prevPos.x === this.wallE.x && prevPos.y === this.wallE.y) {
            return -1;
        }

        // Calculate how many trash pieces were collected
        const initialTrash = this.getTotalTrash();
        const collectedTrash = initialTrash - this.trashCount;
        
        // Collect trash
        if (currentCell === this.TRASH) {
            this.grid[this.wallE.y][this.wallE.x] = this.EMPTY;
            this.trashCount--;
            return 10.0;  // Base reward for collecting trash
        }

        // Hit mine
        if (currentCell === this.MINE) {
            return -20;
        }

        // Reach exit with trash multiplier
        if (currentCell === this.EXIT) {
            if (this.trashCount === 0) {
                // All trash collected: multiply reward by total collected trash
                const multiplier = initialTrash;  // Total trash that was in the level
                return 50.0 * multiplier;  // Base exit reward * number of trash collected
            } else {
                return -10;  // Penalty for reaching exit without all trash
            }
        }

        // Small negative reward for each step to encourage efficiency
        return -0.1;
    }

    getTotalTrash() {
        // Helper function to get total trash in the environment
        let total = this.trashCount;
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                if (this.grid[y][x] === this.TRASH) {
                    total++;
                }
            }
        }
        return total;
    }

    isTerminal() {
        const currentCell = this.grid[this.wallE.y][this.wallE.x];
        return (
            currentCell === this.MINE ||
            currentCell === this.EXIT ||  // Remove trash collection condition
            (this.wallE.x === this.evilRobot.x && this.wallE.y === this.evilRobot.y)
        );
    }

    moveEvilRobot() {
        const moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
        let attempts = 0;
        
        while (attempts < 4) {
            const [dy, dx] = moves[this.evilRobotDirection];
            const newY = this.evilRobot.y + dy;
            const newX = this.evilRobot.x + dx;
            
            if (this.isValidMove(newY, newX)) {
                this.evilRobot.y = newY;
                this.evilRobot.x = newX;
                break;
            }
            
            this.evilRobotDirection = (this.evilRobotDirection + 1) % 4;
            attempts++;
        }
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid and heatmap
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                // Draw cell background with state value color
                if (!this.isEditing && this.grid[y][x] !== this.WALL) {
                    const value = this.stateValues[y][x];
                    this.ctx.fillStyle = this.getStateValueColor(value, x, y);
                } else {
                    this.ctx.fillStyle = '#f0f0f0';
                }
                this.ctx.fillRect(x * this.cellSize, y * this.cellSize, this.cellSize, this.cellSize);
                
                // Draw cell border
                this.ctx.strokeStyle = '#ccc';
                this.ctx.strokeRect(x * this.cellSize, y * this.cellSize, this.cellSize, this.cellSize);

                // Draw cell content
                const cell = this.grid[y][x];
                const padding = 5;
                if (cell === this.WALL && this.assets.wall) {
                    this.ctx.drawImage(this.assets.wall, 
                        x * this.cellSize + padding, 
                        y * this.cellSize + padding, 
                        this.cellSize - 2*padding, 
                        this.cellSize - 2*padding);
                } else if (cell === this.TRASH && this.assets.trash) {
                    this.ctx.drawImage(this.assets.trash,
                        x * this.cellSize + padding,
                        y * this.cellSize + padding,
                        this.cellSize - 2*padding,
                        this.cellSize - 2*padding);
                } else if (cell === this.MINE && this.assets.mine) {
                    this.ctx.drawImage(this.assets.mine,
                        x * this.cellSize + padding,
                        y * this.cellSize + padding,
                        this.cellSize - 2*padding,
                        this.cellSize - 2*padding);
                } else if (cell === this.EXIT && this.assets.exit) {
                    this.ctx.drawImage(this.assets.exit,
                        x * this.cellSize + padding,
                        y * this.cellSize + padding,
                        this.cellSize - 2*padding,
                        this.cellSize - 2*padding);
                }
            }
        }

        // Draw Wall-E
        if (this.assets['wall-e']) {
            this.ctx.drawImage(this.assets['wall-e'],
                this.wallE.x * this.cellSize + 5,
                this.wallE.y * this.cellSize + 5,
                this.cellSize - 10,
                this.cellSize - 10);
        }

        // Draw Evil Robot
        if (this.assets['evil-robot']) {
            this.ctx.drawImage(this.assets['evil-robot'],
                this.evilRobot.x * this.cellSize + 5,
                this.evilRobot.y * this.cellSize + 5,
                this.cellSize - 10,
                this.cellSize - 10);
        }
    }

    resetHeatmap() {
        this.stateVisits = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(0));
        this.maxVisits = 1;
        this.stateValues = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(0));
        this.visitedStates = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(false));
        this.minStateValue = -20;
        this.maxStateValue = 50;
    }

    setupGridEditor() {
        this.canvas.addEventListener('click', (e) => {
            if (!this.isEditing) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / this.cellSize);
            const y = Math.floor((e.clientY - rect.top) / this.cellSize);

            if (x >= 0 && x < this.gridSize && y >= 0 && y < this.gridSize) {
                this.handleGridClick(x, y);
            }
        });

        // Add grid editing controls
        this.isEditing = true;
        document.getElementById('saveGridButton').addEventListener('click', () => {
            this.isEditing = false;
            document.getElementById('startButton').disabled = false;
            // Store custom layout
            this.customGrid = this.grid.map(row => [...row]);
            this.customWallE = {...this.wallE};
            this.customEvilRobot = {...this.evilRobot};
        });

        document.getElementById('resetGridButton').addEventListener('click', () => {
            this.isEditing = true;
            document.getElementById('startButton').disabled = true;
            this.customGrid = null;
            this.customWallE = null;
            this.customEvilRobot = null;
            this.reset();
            this.resetHeatmap();
            this.render();
        });

        // Disable start button while editing
        document.getElementById('startButton').disabled = true;
    }

    handleGridClick(x, y) {
        const tool = document.querySelector('input[name="gridTool"]:checked').value;
        
        // Handle unique elements (Wall-E, evil robot, exit)
        if (tool === 'wall-e') {
            // Remove old Wall-E position
            this.wallE = { x, y };
        } else if (tool === 'evil-robot') {
            // Remove old evil robot position
            this.evilRobot = { x, y };
        } else if (tool === 'exit') {
            // Remove old exit
            for (let i = 0; i < this.gridSize; i++) {
                for (let j = 0; j < this.gridSize; j++) {
                    if (this.grid[i][j] === this.EXIT) {
                        this.grid[i][j] = this.EMPTY;
                    }
                }
            }
            this.grid[y][x] = this.EXIT;
        } else {
            // Handle other elements
            const cellContent = this.getCellContent(tool);
            if (cellContent !== null) {
                // Don't allow placing over Wall-E, evil robot, or exit
                if (!(x === this.wallE.x && y === this.wallE.y) && 
                    !(x === this.evilRobot.x && y === this.evilRobot.y) &&
                    this.grid[y][x] !== this.EXIT) {
                    this.grid[y][x] = cellContent;
                    if (cellContent === this.TRASH) {
                        this.updateTrashCount();
                    }
                }
            }
        }
        
        this.render();
    }

    getCellContent(tool) {
        switch (tool) {
            case 'wall': return this.WALL;
            case 'trash': return this.TRASH;
            case 'mine': return this.MINE;
            case 'empty': return this.EMPTY;
            default: return null;
        }
    }

    updateTrashCount() {
        this.trashCount = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (this.grid[i][j] === this.TRASH) {
                    this.trashCount++;
                }
            }
        }
    }

    setAlgorithm(algorithm) {
        this.currentAlgorithm = algorithm;
        // Reset state values and heatmap when algorithm changes
        this.stateValues = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(0));
        this.minStateValue = -20;
        this.maxStateValue = 50;
        this.render();
        console.log('Switched to algorithm:', algorithm);
    }

    updateStateValue(x, y, value) {
        // Smooth the state value updates
        const alpha = 0.1; // Smoothing factor
        this.stateValues[y][x] = (1 - alpha) * this.stateValues[y][x] + alpha * value;
        this.minStateValue = Math.min(this.minStateValue, this.stateValues[y][x]);
        this.maxStateValue = Math.max(this.maxStateValue, this.stateValues[y][x]);
        
        // Normalize state values periodically
        if (Math.abs(this.maxStateValue) > 1000 || Math.abs(this.minStateValue) > 1000) {
            const maxAbs = Math.max(Math.abs(this.maxStateValue), Math.abs(this.minStateValue));
            for (let i = 0; i < this.gridSize; i++) {
                for (let j = 0; j < this.gridSize; j++) {
                    this.stateValues[i][j] /= maxAbs;
                }
            }
            this.minStateValue /= maxAbs;
            this.maxStateValue /= maxAbs;
        }
    }

    enableEditing() {
        this.isEditing = true;
        document.getElementById('startButton').disabled = true;
        this.resetHeatmap();
        this.render();
        console.log('Grid editing enabled');
    }

    disableEditing() {
        this.isEditing = false;
        this.checkInitialization();
        this.currentAlgorithm = document.getElementById('algorithm').value;
        document.getElementById('startButton').disabled = false;
        console.log('Grid editing disabled, using algorithm:', this.currentAlgorithm);
    }

    setMaxSteps(steps) {
        this.maxSteps = steps;
    }

    getStateValueColor(value, x, y) {
        // Only show colors for visited states
        if (!this.visitedStates[y][x]) {
            return '#ffffff'; // White for unvisited states
        }

        // Normalize value between -1 and 1
        const normalizedValue = (value - this.minStateValue) / (this.maxStateValue - this.minStateValue) * 2 - 1;
        
        // Use HSL color space for smoother transitions
        if (normalizedValue < 0) {
            // Negative values: soft pink to white
            // Hue: 350 (pink/red)
            // Saturation: 30-90% (less saturated for better visibility)
            // Lightness: 90-100% (keeping it lighter)
            const saturation = 30 + Math.abs(normalizedValue) * 60;
            const lightness = 100 - Math.abs(normalizedValue) * 10;
            return `hsl(350, ${saturation}%, ${lightness}%)`;
        } else {
            // Positive values: white to vibrant blue
            // Hue: 210 (blue)
            // Saturation: 40-100% (more saturated for positive values)
            // Lightness: 60-90% (darker blue for more prominence)
            const saturation = 40 + normalizedValue * 60;
            const lightness = 90 - normalizedValue * 30;
            return `hsl(210, ${saturation}%, ${lightness}%)`;
        }
    }
}