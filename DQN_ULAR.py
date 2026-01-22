import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import time

# --- 1. Konfigurasi Game dan Warna ---
BLOCK_SIZE = 20
WIDTH = 640
HEIGHT = 480
SPEED = 40 # FPS / Kecepatan simulasi

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)

# --- 2. Model Deep Q-Network (DQN) ---

class Linear_QNet(nn.Module):
    """Jaringan Saraf Tiruan untuk memprediksi Q-values."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# --- 3. Trainer (Pelatih) untuk Model DQN ---

class QTrainer:
    """Mengatur proses pelatihan model DQN menggunakan Optimisasi dan Loss Function."""
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Penanganan batch/single sample
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. Q-value yang diprediksi untuk state saat ini (Q(s, a))
        pred = self.model(state)
        
        # 2. Menghitung Target Q-value (Rumus Bellman: r + gamma * max(Q(s', a'))
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Update target hanya pada aksi yang diambil
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Optimisasi
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# --- 4. Kelas Game Ular (Lingkungan RL) ---

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Inisialisasi Ular dan Makanan
        self.head = [self.w/2, self.h/2]
        self.snake = [self.head, 
                      [self.head[0]-BLOCK_SIZE, self.head[1]],
                      [self.head[0]-2*BLOCK_SIZE, self.head[1]]]
        self.score = 0
        self.direction = 'RIGHT'
        self.food = self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        # Penempatan makanan
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        if [x, y] in self.snake:
            return self._place_food() # Jika makanan muncul di ular, cari tempat lain
        return [x, y]

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        
        # Tabrakan dengan batas
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] > self.h - BLOCK_SIZE or pt[1] < 0:
            return True
        # Tabrakan dengan diri sendiri
        if pt in self.snake[1:]:
            return True
        return False
        
    def _move(self, direction):
        # Pemindahan kepala ular
        x = self.head[0]
        y = self.head[1]
        
        if direction == 'RIGHT':
            x += BLOCK_SIZE
        elif direction == 'LEFT':
            x -= BLOCK_SIZE
        elif direction == 'DOWN':
            y += BLOCK_SIZE
        elif direction == 'UP':
            y -= BLOCK_SIZE
            
        self.head = [x, y]

    def _update_ui(self):
        # Menggambar ulang tampilan Pygame
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _get_state(self):
        # State Representation (11 nilai biner)
        head_x = self.head[0]
        head_y = self.head[1]
        
        # Arah saat ini
        dir_L = self.direction == 'LEFT'
        dir_R = self.direction == 'RIGHT'
        dir_U = self.direction == 'UP'
        dir_D = self.direction == 'DOWN'
        
        # Titik di depan, kanan, dan kiri (RELATIF terhadap arah ular)
        point_straight = [head_x + (dir_R - dir_L) * BLOCK_SIZE, head_y + (dir_D - dir_U) * BLOCK_SIZE]
        point_right = [head_x + (dir_U - dir_D) * BLOCK_SIZE, head_y + (dir_R - dir_L) * BLOCK_SIZE]
        point_left = [head_x + (dir_D - dir_U) * BLOCK_SIZE, head_y + (dir_L - dir_R) * BLOCK_SIZE]

        state = [
            # 1. Bahaya Langsung
            self._is_collision(point_straight),
            self._is_collision(point_right),
            self._is_collision(point_left),

            # 2. Arah Saat Ini
            dir_L, dir_R, dir_U, dir_D,

            # 3. Lokasi Makanan
            self.food[0] < head_x,  # Makanan Kiri
            self.food[0] > head_x,  # Makanan Kanan
            self.food[1] < head_y,  # Makanan Atas
            self.food[1] > head_y   # Makanan Bawah
        ]

        return np.array(state, dtype=int)
        
    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Periksa event (untuk keluar)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Update arah (aksi adalah [Lurus, Kanan, Kiri] relatif)
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]): # Lurus
            new_dir = clock_wise[idx] 
        elif np.array_equal(action, [0, 1, 0]): # Putar Kanan
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action, [0, 0, 1]): # Putar Kiri
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        # 3. Pindahkan ular
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # 4. Inisialisasi Reward dan Game Over
        reward = 0
        game_over = False
        
        # Hukuman (jika menabrak atau bergerak terlalu lama)
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10 
            return self._get_state(), reward, game_over, self.score
            
        # 5. Makan Makanan
        if self.head == self.food:
            self.score += 1
            reward = 10 
            self.food = self._place_food()
        else:
            self.snake.pop() # Hapus ekor
            
        # 6. Update UI dan Clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 7. Mengembalikan hasilnya
        return self._get_state(), reward, game_over, self.score

# --- 5. Agen RL (The Agent) ---

MAX_MEMORY = 100_000 
BATCH_SIZE = 1000   
LR = 0.001          

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Kontrol randomitas
        self.gamma = 0.9  # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        # Input: 11 (State), Hidden: 256, Output: 3 (Action)
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Ambil sampel acak dari memori
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Strategi Epsilon-Greedy
        # Epsilon meluruh seiring jumlah game bertambah
        self.epsilon = 80 - self.n_games 
        final_move = [0, 0, 0] # [Lurus, Kanan, Kiri]
        
        if random.randint(0, 200) < self.epsilon:
            # Eksplorasi (Random)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Eksploitasi (Model Prediction)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

# --- 6. Fungsi Utama Pelatihan ---

def train():
    agent = Agent()
    game = SnakeGameAI()
    
    total_score = 0
    record = 0 # Skor tertinggi

    print("Memulai Pelatihan DQN untuk Game Ular...")
    print("Agen akan belajar secara bertahap.")
    
    # 

    while True:
        # 1. Dapatkan keadaan awal (state_old)
        state_old = game._get_state() 

        # 2. Dapatkan aksi dari agen (Epsilon-Greedy)
        final_move = agent.get_action(state_old)

        # 3. Ambil langkah di game
        state_new, reward, done, score = game.play_step(final_move)

        # 4. Latih memori singkat (short memory)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Ingat (simpan pengalaman di replay memory)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 6. Reset game dan perbarui statistik
            game.reset()
            agent.n_games += 1
            agent.train_long_memory() # Latih dari replay memory

            if score > record:
                record = score
                # Opsional: Simpan model terbaik di sini
                # torch.save(agent.model.state_dict(), 'model/model_best.pth')

            total_score += score
            mean_score = total_score / agent.n_games
            
            print(f'Game: {agent.n_games} | Score: {score} | Record: {record} | Mean Score: {mean_score:.2f} | Epsilon: {agent.epsilon}')

if __name__ == '__main__':
    train()
