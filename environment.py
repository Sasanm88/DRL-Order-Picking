import numpy as np
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
            
class Env(object):
    def __init__(self, n_row, n_aisle, arrival_rate, capacity, depot, inter_aisle_distance, pick_up_time, drop_off_time, alpha):
        self.n_row = n_row
        self.n_col = 3 * n_aisle
        self.n_aisle = n_aisle
        self.arrival_rate = arrival_rate     
        self.depot_aisle = depot
        self.depot_col = 3 * self.depot_aisle +1
        self.alpha = alpha 
        
        self.capacity = capacity
        self.inter_aisle_distance = inter_aisle_distance
        self.pick_up_time = pick_up_time
        self.drop_off_time = drop_off_time
        
        self.ups = [i for i in range(2*self.n_aisle) if i%2==0]
        self.downs = [i for i in range(2*self.n_aisle) if i%2==1]
        self.up_distances = {}
        self.down_distances = {}
        for row in range(self.n_row):
            for col in range(self.n_aisle):
                self.up_distances[row, col] = torch.zeros(self.n_row-2, self.n_aisle)
                self.down_distances[row, col] = torch.zeros(self.n_row-2, self.n_aisle)
                for i in range(1, self.n_row-1):
                    for j in range(self.n_aisle):
                        if col == j:
                            self.up_distances[row, col][i-1,j] = abs(row-i) + 0.5*(row==i)
                            self.down_distances[row, col][i-1,j] = abs(row-i) + 0.5*(row==i)
                        else:
                            self.up_distances[row, col][i-1,j] = 3*abs(col-j) + row+i
                            self.down_distances[row, col][i-1,j] = 3*abs(col-j) + 2*self.n_row-row-i
    
    def calculate_stateP(self):   
        if self.picker_row == 0:
            temp2 = self.prizes_aisle[1:-1] /self.down_distances[self.picker_row, self.picker_aisle]
            self.stateP[self.ups] = 0
            self.stateP[self.downs] = torch.sum(temp2, axis=0)
        elif self.picker_row == self.n_row-1:
            temp1 = self.prizes_aisle[1:-1] /self.up_distances[self.picker_row, self.picker_aisle]
            self.stateP[self.ups] = torch.sum(temp1, axis=0)
            self.stateP[self.downs] = 0
        else:
            temp1 = self.prizes_aisle[1:-1] /self.up_distances[self.picker_row, self.picker_aisle]
            temp2 = self.prizes_aisle[1:-1] /self.down_distances[self.picker_row, self.picker_aisle]
            self.stateP[self.downs] = torch.sum(temp2, axis=0)           
            self.stateP[self.ups] = torch.sum(temp1, axis=0)           
            self.stateP[self.picker_aisle*2] = torch.sum(temp1[:self.picker_row,self.picker_aisle])
            self.stateP[self.picker_aisle*2+1] = torch.sum(temp2[self.picker_row-1:,self.picker_aisle])
        
    def reset(self):
        self.picker_row = self.n_row - 1
        self.picker_aisle = self.depot_aisle
        self.prizes_aisle = torch.zeros(self.n_row, self.n_aisle, dtype = torch.float32, device = device)
        self.start_first_prize()
        self.stateP = torch.zeros(2*self.n_aisle, dtype = torch.float32, device = device)  
        self.calculate_stateP()
        self.stateM = torch.tensor([1, 2*self.depot_aisle, 2*self.depot_aisle+1, self.capacity], dtype = torch.float32, device = device)   
        allowed_actions, allowed_actions_tensor = self.available_actions()
        return self.stateM.clone().detach(), self.stateP.clone().detach(), allowed_actions, allowed_actions_tensor

        
    def available_actions(self):
        allowed_actions = torch.zeros(5)
        if self.stateM[3] == 0:
            if self.stateM[0] < 1:
                allowed_actions[4] = 1
                actions = [4]
            else:
                if self.picker_aisle < self.depot_aisle:
                    allowed_actions[1]
                    actions = [1]
                elif self.picker_aisle > self.depot_aisle:
                    allowed_actions[2]
                    actions = [2]
                else:
                    allowed_actions[0]
                    actions = [0]
        else:
            actions = [0]
            allowed_actions[0] = 1

            if self.stateM[0] != 0:
                if self.stateM[1] == 0:
                    actions.append(1)
                    allowed_actions[1] = 1
                elif self.stateM[1] == 2*(self.n_aisle-1):
                    actions.append(2)
                    allowed_actions[2] = 1
                else:
                    actions.extend([1,2])
                    allowed_actions[1] = 1
                    allowed_actions[2] = 1

                if self.stateM[0] == -1:
                    actions.append(4)
                    allowed_actions[4] = 1
                else:
                    actions.append(3)
                    allowed_actions[3] = 1
            else:
                actions.extend([3,4])
                allowed_actions[3] = 1
                allowed_actions[4] = 1
            
        return actions, allowed_actions

    def new_orders(self, action, t):
        k = np.random.poisson(t*self.arrival_rate)  
        change_of_prize_state = False
        while k > 0:
            row = random.randint(1, self.n_row-2)
            aisle = random.randint(0, self.n_aisle-1)
#             aisle = random.choices(self.prize_aisle, weights = self.weights)[0]
            self.prizes_aisle[row, aisle] += 1
            k -= 1
            if aisle == self.picker_aisle:
                if action == 3:
                    if row > self.picker_row:
                        change_of_prize_state = True
                elif action == 4:
                    if row < self.picker_row:
                        change_of_prize_state = True
        return change_of_prize_state
            
            
    def start_first_prize(self):
        if random.random() < 0.5:
            return
        k = random.randint(1,10)
        while k > 0:
            k -= 1
            row = random.randint(1, self.n_row-2)
            aisle = random.randint(0, self.n_aisle-1)
            self.prizes_aisle[row, aisle] += 1
                        
    def step(self, action):
        #action: 0=stay, 1=right, 2=left, 3=up, 4=down   
        reward = torch.tensor(0.0, device = device)
        if action == 1:
            self.picker_aisle += 1
            self.stateM[1] += 2
            self.stateM[2] += 2
            reward -= 3
            self.new_orders(action, self.inter_aisle_distance)
        elif action == 2:
            
            self.picker_aisle -= 1
            self.stateM[1] -= 2
            self.stateM[2] -= 2
            self.new_orders(action, self.inter_aisle_distance)
            reward -= 3
            
        elif action == 3:
            any_rewards_picked = False
            if self.stateM[0] == 1:
                self.stateM[0] = 0
                self.picker_row -= 1
                reward -= 1
                self.new_orders(action, 1)
                
                while True:
                    if self.stateM[3] > 0 and self.prizes_aisle[self.picker_row, self.picker_aisle] > 0:
                        reward += self.n_row + self.n_col
                        self.stateM[3] -= 1
                        self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                        self.new_orders(action, self.pick_up_time)
                        any_rewards_picked = True
                    else:
                        break
            change_of_prize_state = False
            if not any_rewards_picked and self.stateM[0] == 0: 
                
                while not any_rewards_picked and self.picker_row > 1:
                    if change_of_prize_state:
                        break
                    reward -= 1
                    self.picker_row -= 1
                    change_of_prize_state = self.new_orders(action, 1) or change_of_prize_state

                    while True:
                        if self.stateM[3] == 0 or self.prizes_aisle[self.picker_row, self.picker_aisle] == 0:
                            break
                        else:
                            any_rewards_picked = True
                            reward += self.n_row + self.n_col
                            self.stateM[3] -= 1
                            self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                            change_of_prize_state = self.new_orders(action, self.pick_up_time) or change_of_prize_state
            if self.picker_row == 1 and not any_rewards_picked and not change_of_prize_state:
                self.stateM[0] -= 1
                self.picker_row = 0 
                reward -= 1
                self.new_orders(action, 1)
                
                        
        elif action == 4:
            any_rewards_picked = False
            if self.stateM[0] == -1:
                self.stateM[0] = 0
                self.picker_row += 1
                reward -= 1
                self.new_orders(action, 1)
                
                while True:
                    if self.stateM[3] > 0 and self.prizes_aisle[self.picker_row, self.picker_aisle] > 0:
                        reward += self.n_row + self.n_col
                        self.stateM[3] -= 1
                        self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                        self.new_orders(action, self.pick_up_time)
                        any_rewards_picked = True
                    else:
                        break
                        
            change_of_prize_state = False
            if not any_rewards_picked and self.stateM[0] == 0: 
                while not any_rewards_picked and self.picker_row < self.n_row-2:
                    if change_of_prize_state: 
                        break
                    reward -= 1
                    self.picker_row += 1
                    change_of_prize_state = self.new_orders(action, 1) or change_of_prize_state

                    while True:
                        if self.stateM[3] == 0 or self.prizes_aisle[self.picker_row, self.picker_aisle] == 0:
                            break
                        else:
                            any_rewards_picked = True
                            reward += self.n_row + self.n_col
                            self.stateM[3] -= 1
                            self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                            change_of_prize_state = self.new_orders(action, self.pick_up_time) or change_of_prize_state
            if self.picker_row == self.n_row-2 and not any_rewards_picked and not change_of_prize_state:
                self.stateM[0] = 1
                self.picker_row += 1 
                reward -= 1
                self.new_orders(action, 1)

        
        if action == 0:
            if self.picker_row == self.n_row-1 and self.picker_aisle == self.depot_aisle and self.stateM[3] < self.capacity:
                released_items = self.capacity - self.stateM[3]
                reward += (self.n_row + self.n_col) * released_items * self.alpha
                self.stateM[3] = self.capacity 
#                 for i in range(int(released_items)):
                self.new_orders(action, self.drop_off_time*int(released_items))

            else:
                reward -= 1
                self.new_orders(action, 1)
                
        self.calculate_stateP()
        allowed_actions, allowed_actions_tensor = self.available_actions()
        return self.stateM.clone().detach(), self.stateP.clone().detach(), reward*0.1, allowed_actions, allowed_actions_tensor
        
    def reset_show(self, history, minimum_pick_list):
        self.minimum_pick_list = minimum_pick_list
        self.order_count = 0
        self.start_from_depot = True
        
        self.use_history = len(history) > 0
        self.history = history

        self.time_step = 0
        self.picker_row = self.n_row - 1
        self.picker_aisle = self.depot_aisle
        self.picker_col = self.depot_col
        self.room = self.capacity
        self.prizes_col = torch.zeros(self.n_row, self.n_col, dtype = torch.float32, device = device)       
        self.prizes_aisle = torch.zeros(self.n_row, self.n_aisle, dtype = torch.float32, device = device)
        self.start_first_prize_show()
        self.stateP = torch.zeros(2*self.n_aisle, dtype = torch.float32, device = device)  
        self.calculate_stateP()
        self.stateM = torch.tensor([1, 2*self.depot_aisle, 2*self.depot_aisle+1, self.capacity], dtype = torch.float32, device = device)   
        if self.start_from_depot and self.order_count < self.minimum_pick_list: 
            allowed_actions_tensor = torch.zeros(5)
            allowed_actions = [0]
            allowed_actions_tensor[0] = 1
        else:
            allowed_actions, allowed_actions_tensor = self.available_actions()
        return self.stateM.clone().detach(), self.stateP.clone().detach(), allowed_actions, allowed_actions_tensor, torch.tensor([self.picker_row, self.picker_col, self.capacity]), self.prizes_col.clone().detach()
    
    def start_first_prize_show(self):
        if self.use_history:
            row, aisle = self.history[0][0]
        else:
            row = random.randint(1, self.n_row-2)
#             aisle = random.choices(self.prize_aisle, weights = self.weights)[0]
            aisle = random.randint(0, self.n_aisle-1)
            self.history.append([(row, aisle)])
        col = 3*aisle+1
        self.prizes_col[row, col] += 1
        self.prizes_aisle[row, aisle] += 1
        self.order_count += 1
        
    def new_orders_show(self, action):
        change_of_prize_state = False
        if self.use_history:
            for (row, aisle) in self.history[self.time_step]:
                col = 3*aisle+1
                self.prizes_col[row, col] += 1
                self.prizes_aisle[row, aisle] += 1
                self.order_count += 1
                if aisle == self.picker_aisle:
                    if action == 3:
                        if row > self.picker_row:
                            change_of_prize_state = True
                    elif action == 4:
                        if row < self.picker_row:
                            change_of_prize_state = True
        else:
            k = np.random.poisson(self.arrival_rate)  
            arrivals = []
            while k > 0:
                row = random.randint(1, self.n_row-2)
#                 aisle = random.choices(self.prize_aisle, weights = self.weights)[0]
                aisle = random.randint(0, self.n_aisle-1) 
                arrivals.append((row, aisle))
                col = 3*aisle+1
                self.prizes_col[row, col] += 1
                self.prizes_aisle[row, aisle] += 1
                k -= 1
                self.order_count += 1
                if aisle == self.picker_aisle:
                    if action == 3:
                        if row > self.picker_row:
                            change_of_prize_state = True
                    elif action == 4:
                        if row < self.picker_row:
                            change_of_prize_state = True
            self.history.append(arrivals)
        return change_of_prize_state
    
    def step_show(self, action):
        #action: 0=stay, 1=right, 2=left, 3=up, 4=down   
        all_prizes = []
        all_picker_locs = [] 
        reward = torch.tensor(0.0, device = device)
        if action == 1:
            self.start_from_depot = False
            self.picker_aisle += 1
            self.stateM[1] += 2
            self.stateM[2] += 2
            for i in range(self.inter_aisle_distance): 
                self.time_step += 1
                reward -= 1                
                self.new_orders_show(action)
                if i < 3:
                    self.picker_col += 1
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))                
                    all_prizes.append(self.prizes_col.clone().detach())
        elif action == 2:
            self.start_from_depot = False
            self.picker_aisle -= 1
            self.stateM[1] -= 2
            self.stateM[2] -= 2
            for i in range(self.inter_aisle_distance): 
                self.time_step += 1
                reward -= 1
                self.new_orders_show(action)
                if i < 3:
                    self.picker_col -= 1
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
        elif action == 3:
            self.start_from_depot = False
            any_rewards_picked = False
            if self.stateM[0] == 1:
                self.time_step += 1
                self.stateM[0] = 0
                self.picker_row -= 1
                reward -= 1
                self.new_orders_show(action)
                
                while True:
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
                    if self.room > 0 and self.prizes_col[self.picker_row, self.picker_col] > 0:
                        reward += self.n_row
                        self.room -= 1
                        self.stateM[3] -= 1
                        self.order_count -= 1
                        self.prizes_col[self.picker_row, self.picker_col] -= 1
                        self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                        for i in range(self.pick_up_time):
                            self.time_step += 1
                            self.new_orders_show(action)
                        any_rewards_picked = True
                    else:
                        break
            change_of_prize_state = False
            if not any_rewards_picked and self.stateM[0] == 0: 
                
                while not any_rewards_picked and self.picker_row > 1:
                    if change_of_prize_state:
                        break
                    reward -= 1
                    self.picker_row -= 1
                    self.time_step += 1
                    change_of_prize_state = self.new_orders_show(action) or change_of_prize_state

                    while True:
                        all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                        all_prizes.append(self.prizes_col.clone().detach())
                        if self.room == 0 or self.prizes_col[self.picker_row, self.picker_col] == 0:
                            break
                        else:
                            any_rewards_picked = True
                            reward += self.n_row
                            self.room -= 1
                            self.stateM[3] -= 1
                            self.order_count -= 1
                            self.prizes_col[self.picker_row, self.picker_col] -= 1
                            self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                            for i in range(self.pick_up_time):
                                self.time_step += 1
                                change_of_prize_state = self.new_orders_show(action) or change_of_prize_state
            if self.picker_row == 1 and not any_rewards_picked and not change_of_prize_state:
                self.stateM[0] -= 1
                self.picker_row = 0 
                reward -= 1
                self.time_step += 1
                self.new_orders_show(action)
                all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                all_prizes.append(self.prizes_col.clone().detach())
                
                        
        elif action == 4:
            self.start_from_depot = False
            any_rewards_picked = False
            if self.stateM[0] == -1:
                self.stateM[0] = 0
                self.picker_row += 1
                reward -= 1
                self.time_step += 1
                self.new_orders_show(action)
                
                while True:
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
                    if self.room > 0 and self.prizes_col[self.picker_row, self.picker_col] > 0:
                        reward += self.n_row
                        self.room -= 1
                        self.stateM[3] -= 1
                        self.order_count -= 1
                        self.prizes_col[self.picker_row, self.picker_col] -= 1
                        self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                        for i in range(self.pick_up_time):
                            self.time_step += 1
                            self.new_orders_show(action)
                        any_rewards_picked = True
                    else:
                        break
                        
            change_of_prize_state = False
            if not any_rewards_picked and self.stateM[0] == 0: 
                while not any_rewards_picked and self.picker_row < self.n_row-2:
                    if change_of_prize_state: 
                        break
                    reward -= 1
                    self.picker_row += 1
                    self.time_step += 1
                    change_of_prize_state = self.new_orders_show(action) or change_of_prize_state

                    while True:
                        all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                        all_prizes.append(self.prizes_col.clone().detach())
                        if self.room == 0 or self.prizes_col[self.picker_row, self.picker_col] == 0:
                            break
                        else:
                            any_rewards_picked = True
                            reward += self.n_row
                            self.room -= 1
                            self.stateM[3] -= 1
                            self.order_count -= 1
                            self.prizes_col[self.picker_row, self.picker_col] -= 1
                            self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                            for i in range(self.pick_up_time):
                                self.time_step += 1
                                change_of_prize_state = self.new_orders_show(action) or change_of_prize_state
            if self.picker_row == self.n_row-2 and not any_rewards_picked and not change_of_prize_state:
                self.stateM[0] = 1
                self.picker_row += 1 
                reward -= 1
                self.time_step += 1
                self.new_orders_show(action)
                all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                all_prizes.append(self.prizes_col.clone().detach())

        
        if action == 0:
            if self.picker_row == self.n_row-1 and self.picker_col == self.depot_col and self.room < self.capacity:
                self.start_from_depot = True
                released_items = self.capacity - self.room
#                 reward += released_items 
                reward += (self.n_aisle) * released_items 
                self.stateM[3] = self.capacity 
                for i in range(released_items):
                    for j in range(self.drop_off_time):
                        self.time_step += 1
                        self.new_orders_show(action)
                    self.room += 1
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
            else:
                if self.picker_row == self.n_row-1 and self.picker_col == self.depot_col:
                    self.start_from_depot = True
                else:
                    self.start_from_depot = False
                reward -= 1
                self.time_step += 1
                self.new_orders_show(action)
                all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                all_prizes.append(self.prizes_col.clone().detach())
                
        self.calculate_stateP()
        
        if self.start_from_depot and self.order_count < self.minimum_pick_list: 
            allowed_actions_tensor = torch.zeros(5)
            allowed_actions = [0]
            allowed_actions_tensor[0] = 1
        else:
            allowed_actions, allowed_actions_tensor = self.available_actions()
        final_state = self.use_history and self.time_step == len(self.history)-1
        return self.stateM.clone().detach(), self.stateP.clone().detach(), reward, allowed_actions, allowed_actions_tensor, all_picker_locs, all_prizes, final_state
    
    
    
    def reset_test(self, history, minimum_pick_list):   
        self.minimum_pick_list = minimum_pick_list
        self.order_count = 0
        self.start_from_depot = True
        
        self.use_history = len(history) > 0
        self.history = history
        self.orders_in_aisles = [[[] for i in range(self.n_aisle)] for j in range(self.n_row)]
        self.orders_in_picker = []
        self.delivered_orders = []
        self.time_step = 0
        self.picker_row = self.n_row - 1
        self.picker_aisle = self.depot_aisle
        self.picker_col = self.depot_col
        self.room = self.capacity
        self.prizes_col = torch.zeros(self.n_row, self.n_col, dtype = torch.float32, device = device)       
        self.prizes_aisle = torch.zeros(self.n_row, self.n_aisle, dtype = torch.float32, device = device)
#         self.start_first_prize_test()
        self.stateP = torch.zeros(2*self.n_aisle, dtype = torch.float32, device = device)  
        self.calculate_stateP()
        self.stateM = torch.tensor([1, 2*self.depot_aisle, 2*self.depot_aisle+1, self.capacity], dtype = torch.float32, device = device)   
        if self.start_from_depot and self.order_count < self.minimum_pick_list: 
            allowed_actions_tensor = torch.zeros(5)
            allowed_actions = [0]
            allowed_actions_tensor[0] = 1
        else:
            allowed_actions, allowed_actions_tensor = self.available_actions()
            
        self.new_order_arrived = False
        self.new_order_arrived_before = False
        self.last_action = 0
        self.two_last_action = 0
        self.trapped_in_loop = False
        self.manual_steps = 0
        self.all_steps = 0
        return self.stateM.clone().detach(), self.stateP.clone().detach(), allowed_actions, allowed_actions_tensor, torch.tensor([self.picker_row, self.picker_col, self.capacity]), self.prizes_col.clone().detach()
    
    def start_first_prize_test(self):
        if self.use_history:
            row, aisle = self.history[0][0]
        else:
            row = random.randint(1, self.n_row-2)
            aisle = random.randint(0, self.n_aisle-1) #random.choices(self.prize_aisle, weights = self.weights)[0]
            self.history.append([(row, aisle)])
        col = 3*aisle+1
        self.prizes_col[row, col] += 1
        self.prizes_aisle[row, aisle] += 1
        self.orders_in_aisles[row][aisle].append(Order(self.time_step))
        self.order_count += 1
        
    def new_orders_test(self, action):
        change_of_prize_state = False
        if self.use_history:
            if len(self.history) <= self.time_step:
                return False
            for (row, aisle) in self.history[self.time_step]:
                col = 3*aisle+1
                self.prizes_col[row, col] += 1
                self.prizes_aisle[row, aisle] += 1
                self.orders_in_aisles[row][aisle].append(Order(self.time_step))
                self.order_count += 1
                self.new_order_arrived = True
                if aisle == self.picker_aisle:
                    if action == 3:
                        if row > self.picker_row:
                            change_of_prize_state = True
                    elif action == 4:
                        if row < self.picker_row:
                            change_of_prize_state = True
        else:
            k = np.random.poisson(self.arrival_rate)  
            arrivals = []
            while k > 0:
                row = random.randint(1, self.n_row-2)
                aisle = random.randint(0, self.n_aisle-1)
#                 aisle = random.choices(self.prize_aisle, weights = self.weights)[0]
                arrivals.append((row, aisle))
                col = 3*aisle+1
                self.prizes_col[row, col] += 1
                self.prizes_aisle[row, aisle] += 1
                self.orders_in_aisles[row][aisle].append(Order(self.time_step))
                k -= 1
                self.order_count += 1
                if aisle == self.picker_aisle:
                    if action == 3:
                        if row > self.picker_row:
                            change_of_prize_state = True
                    elif action == 4:
                        if row < self.picker_row:
                            change_of_prize_state = True
            self.history.append(arrivals)
        return change_of_prize_state
    
    def step_test(self, action):
        #action: 0=stay, 1=right, 2=left, 3=up, 4=down   
        
        
#             print(f'action = {action} , still trapped = {self.trapped_in_loop}')
        if not self.trapped_in_loop: 
            
            if not self.new_order_arrived: 
                if self.last_action == 1 and action == 2:
                    self.trapped_in_loop = True
                if self.last_action == 2 and action == 1:
                    self.trapped_in_loop = True
                if self.last_action == 3 and action == 4:
                    self.trapped_in_loop = True
                if self.last_action == 4 and action == 3:
                    self.trapped_in_loop = True
            if self.order_count > 6 and self.last_action == 0 and action == 0:
                self.trapped_in_loop = True
#                 print(f'action 1 = {self.two_last_action} , actio 2 = {self.last_action}')

        if self.trapped_in_loop: 
            self.manual_steps += 1
            action, self.trapped_in_loop = self.manual_take_over()
            
        self.last_action = action
        self.all_steps += 1
        
            
#         self.two_last_action = self.last_action
#         self.last_action = action       
#         if self.trapped_in_loop:
#             go to manual mode until the first pick up

        self.new_order_arrived = False
        
        all_prizes = []
        all_picker_locs = [] 
        reward = torch.tensor(0.0, device = device)
        if action == 1:
            self.start_from_depot = False
            self.picker_aisle += 1
            self.stateM[1] += 2
            self.stateM[2] += 2
            for order in self.orders_in_picker:
                order.distance += self.inter_aisle_distance
            for i in range(self.inter_aisle_distance): 
                self.time_step += 1
                reward -= 1
                self.new_orders_test(action)
                if i<3:
                    self.picker_col += 1
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))

                    all_prizes.append(self.prizes_col.clone().detach())
        elif action == 2:
            self.start_from_depot = False
            self.picker_aisle -= 1
            self.stateM[1] -= 2
            self.stateM[2] -= 2
            for order in self.orders_in_picker:
                order.distance += self.inter_aisle_distance
            for i in range(self.inter_aisle_distance): 
                self.time_step += 1
                reward -= 1
                self.new_orders_test(action)
                if i<3:
                    self.picker_col -= 1
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))

                    all_prizes.append(self.prizes_col.clone().detach())
                
        elif action == 3:
            self.start_from_depot = False
            any_rewards_picked = False
            if self.stateM[0] == 1:
                self.time_step += 1
                self.stateM[0] = 0
                self.picker_row -= 1
                reward -= 1
                for order in self.orders_in_picker:
                    order.distance += 1
                self.new_orders_test(action)
                
                while True:
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
                    if self.room > 0 and self.prizes_col[self.picker_row, self.picker_col] > 0:
                        reward += self.n_row
                        self.room -= 1
                        self.stateM[3] -= 1
                        self.order_count -= 1
                        self.orders_in_picker.append(self.orders_in_aisles[self.picker_row][self.picker_aisle].pop())
                        self.prizes_col[self.picker_row, self.picker_col] -= 1
                        self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                        for i in range(self.pick_up_time):
                            self.time_step += 1
                            self.new_orders_test(action)
                        any_rewards_picked = True
                        
                    else:
                        break
            change_of_prize_state = False
            if not any_rewards_picked and self.stateM[0] == 0:                 
                while not any_rewards_picked and self.picker_row > 1:
                    if change_of_prize_state:
                        break
                    reward -= 1
                    self.picker_row -= 1
                    for order in self.orders_in_picker:
                        order.distance += 1
                    self.time_step += 1
                    change_of_prize_state = self.new_orders_test(action) or change_of_prize_state

                    while True:
                        all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                        all_prizes.append(self.prizes_col.clone().detach())
                        if self.room == 0 or self.prizes_col[self.picker_row, self.picker_col] == 0:
                            break
                        else:
                            any_rewards_picked = True
                            reward += self.n_row
                            self.room -= 1
                            self.stateM[3] -= 1
                            self.order_count -= 1
                            self.orders_in_picker.append(self.orders_in_aisles[self.picker_row][self.picker_aisle].pop())
                            self.prizes_col[self.picker_row, self.picker_col] -= 1
                            self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                            for i in range(self.pick_up_time):
                                self.time_step += 1
                                change_of_prize_state = self.new_orders_test(action) or change_of_prize_state
            if self.picker_row == 1 and not any_rewards_picked and not change_of_prize_state:
                self.stateM[0] -= 1
                self.picker_row = 0 
                for order in self.orders_in_picker:
                    order.distance += 1
                reward -= 1
                self.time_step += 1
                self.new_orders_test(action)
                all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                all_prizes.append(self.prizes_col.clone().detach())
                
                        
        elif action == 4:
            self.start_from_depot = False
            any_rewards_picked = False
            if self.stateM[0] == -1:
                self.stateM[0] = 0
                self.picker_row += 1
                for order in self.orders_in_picker:
                    order.distance += 1
                reward -= 1
                self.time_step += 1
                self.new_orders_test(action)
                
                while True:
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
                    if self.room > 0 and self.prizes_col[self.picker_row, self.picker_col] > 0:
                        reward += self.n_row
                        self.room -= 1
                        self.stateM[3] -= 1
                        self.order_count -= 1
                        self.orders_in_picker.append(self.orders_in_aisles[self.picker_row][self.picker_aisle].pop())
                        self.prizes_col[self.picker_row, self.picker_col] -= 1
                        self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                        for i in range(self.pick_up_time):
                            self.time_step += 1
                            self.new_orders_test(action)
                        any_rewards_picked = True
                    else:
                        break
                        
            change_of_prize_state = False
            if not any_rewards_picked and self.stateM[0] == 0: 
                while not any_rewards_picked and self.picker_row < self.n_row-2:
                    if change_of_prize_state: 
                        break
                    reward -= 1
                    self.picker_row += 1
                    for order in self.orders_in_picker:
                        order.distance += 1
                    self.time_step += 1
                    change_of_prize_state = self.new_orders_test(action) or change_of_prize_state

                    while True:
                        all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                        all_prizes.append(self.prizes_col.clone().detach())
                        if self.room == 0 or self.prizes_col[self.picker_row, self.picker_col] == 0:
                            break
                        else:
                            any_rewards_picked = True
                            reward += self.n_row
                            self.room -= 1
                            self.stateM[3] -= 1
                            self.order_count -= 1
                            self.orders_in_picker.append(self.orders_in_aisles[self.picker_row][self.picker_aisle].pop())
                            self.prizes_col[self.picker_row, self.picker_col] -= 1
                            self.prizes_aisle[self.picker_row, self.picker_aisle] -= 1
                            for i in range(self.pick_up_time):
                                self.time_step += 1
                                change_of_prize_state = self.new_orders_test(action) or change_of_prize_state
            if self.picker_row == self.n_row-2 and not any_rewards_picked and not change_of_prize_state:
                self.stateM[0] = 1
                self.picker_row += 1 
                for order in self.orders_in_picker:
                    order.distance += 1
                reward -= 1
                self.time_step += 1
                self.new_orders_test(action)
                all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                all_prizes.append(self.prizes_col.clone().detach())

        
        if action == 0:
            if self.picker_row == self.n_row-1 and self.picker_col == self.depot_col and self.room < self.capacity:
                self.start_from_depot = True
                released_items = self.capacity - self.room
                reward += released_items 
                self.stateM[3] = self.capacity 
                for i in range(released_items):
                    for j in range(self.drop_off_time):
                        self.time_step += 1
                        self.new_orders_test(action)
                    order = self.orders_in_picker.pop()
                    order.delivery_time = self.time_step
                    self.delivered_orders.append(order)
                    
                    self.room += 1
                    all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                    all_prizes.append(self.prizes_col.clone().detach())
            else:
                if self.picker_row == self.n_row-1 and self.picker_col == self.depot_col:
                    self.start_from_depot = True
                else:
                    self.start_from_depot = False
                reward -= 1
                self.time_step += 1
                self.new_orders_test(action)
                all_picker_locs.append(torch.tensor([self.picker_row, self.picker_col, self.room]))
                all_prizes.append(self.prizes_col.clone().detach())
                
        self.calculate_stateP()
        if self.start_from_depot and self.order_count < self.minimum_pick_list: 
            allowed_actions_tensor = torch.zeros(5)
            allowed_actions = [0]
            allowed_actions_tensor[0] = 1
        else:
            allowed_actions, allowed_actions_tensor = self.available_action_test()
        final_state = self.use_history and self.time_step >= len(self.history)-1
        return self.stateM.clone().detach(), self.stateP.clone().detach(), reward, allowed_actions, allowed_actions_tensor, all_picker_locs, all_prizes, final_state
    
    
    def available_action_test(self):
        allowed_actions = torch.zeros(5)
        if self.stateM[3] == 0:
            if self.stateM[0] < 1:
                allowed_actions[4] = 1
                actions = [4]
            else:
                if self.picker_aisle < self.depot_aisle:
                    allowed_actions[1]
                    actions = [1]
                elif self.picker_aisle > self.depot_aisle:
                    allowed_actions[2]
                    actions = [2]
                else:
                    allowed_actions[0]
                    actions = [0]
        elif self.order_count == 0:
            actions = [0]
            allowed_actions[0] = 1
            if self.stateM[0] < 1:
                allowed_actions[4] = 1
                actions.append(4)
            else:
                if self.picker_aisle < self.depot_aisle:
                    allowed_actions[1]
                    actions.append(1)
                elif self.picker_aisle > self.depot_aisle:
                    allowed_actions[2]
                    actions.append(2)
                else:
                    allowed_actions[0]
                    actions = [0]
                    
        else:
                        
            actions = [0]
            allowed_actions[0] = 1

            if self.stateM[0] != 0:
                if self.stateM[1] == 0:
                    actions.append(1)
                    allowed_actions[1] = 1
                elif self.stateM[1] == 2*(self.n_aisle-1):
                    actions.append(2)
                    allowed_actions[2] = 1
                else:
                    actions.extend([1,2])
                    allowed_actions[1] = 1
                    allowed_actions[2] = 1

                if self.stateM[0] == -1:
                    actions.append(4)
                    allowed_actions[4] = 1
                else:
                    actions.append(3)
                    allowed_actions[3] = 1
            else:
                actions.extend([3,4])
                allowed_actions[3] = 1
                allowed_actions[4] = 1
            
        return actions, allowed_actions
    
    def manual_take_over(self):
        if self.picker_row == 0: 
            best_aisle = torch.argmax(torch.sum(self.prizes_aisle, 0))
            if self.picker_aisle < best_aisle: 
                return 1, True
            elif self.picker_aisle > best_aisle: 
                return 2, True
            else:
                return 4, False
        elif self.picker_row == self.n_row-1: 
            best_aisle = torch.argmax(torch.sum(self.prizes_aisle, 0))
            if self.picker_aisle < best_aisle: 
                return 1, True
            elif self.picker_aisle > best_aisle: 
                return 2, True
            else:
                return 3, False
        else: 
            up_reward = torch.sum(self.prizes_aisle[:self.picker_row, self.picker_aisle])
            down_reward = torch.sum(self.prizes_aisle[self.picker_row:, self.picker_aisle])
            if up_reward > 0 or down_reward > 0:
                if up_reward > down_reward: 
                    return 3, False
                else:
                    return 4, False
            else:
                if self.picker_row < 7: 
                    return 3, False
                else:
                    return 4, False
            
        
    
class Order(object):
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time
        self.distance = 0
        self.delivery_time = 0