from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
import random
import time
start = time.time()

class LanAgent(Agent):
    def __init__(self, unique_id, model, state, pos):
        super().__init__(unique_id, model)
        self.state = state
        self.pos = pos

    def calculate_pressure(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,  
            include_center=False
        )
        return sum(neighbor.state for neighbor in neighbors)

    def step(self):
        pressure = self.calculate_pressure()

        S_a, S_b, S_c = 3, 9, 14

        old_state = self.state

        if self.state == 0:
            self.state = 1 if pressure > S_b else 0
        elif self.state == 1:
            if pressure < S_b:
                self.state = 0
            elif S_b <= pressure <= S_c:
                self.state = 1
            else:
                self.state = 2
        elif self.state == 2:
            if pressure <= S_a:
                self.state = 0
            elif S_a < pressure <= S_b:
                self.state = 2
            else:
                self.state = 1

        print(
            f"Agent {self.unique_id} at {self.pos}: "
            f"Old State={old_state}, Pressure={pressure}, New State={self.state}"
        )
   

class LanModel(Model):
    def __init__(self, N, width, height):
        if N > width * height:
            raise ValueError("Number of agents exceeds grid capacity.")
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        probabilities = [0.023, 0.067, 0.909]

        agent_id = 0
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if agent_id < self.num_agents:
                    state = random.choices([0, 1, 2], weights=probabilities, k=1)[0]
                    agent = LanAgent(agent_id, self, state, (x, y))
                    self.schedule.add(agent)
                    self.grid.place_agent(agent, (x, y))
                    agent_id += 1

    def step(self):
        self.schedule.step()


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}
    if agent.state == 0:
        portrayal["Color"] = "red"  
    elif agent.state == 1:
        portrayal["Color"] = "green"  
    elif agent.state == 2:
        portrayal["Color"] = "blue"
    portrayal["Layer"] = 0
    return portrayal


end = time.time()
print('runtime: %s Seconds'%(end-start))

model_description = "This is a model simulating language state changes based on language data from Hong Kong."
grid = CanvasGrid(agent_portrayal, 105, 64, 525, 320)  
server = ModularServer(
    LanModel,
    [grid],
    "Language Model",
    {"N": 6720, "width": 105, "height": 64}  
)
server.description = model_description
server.port = 8080  
server.launch()
