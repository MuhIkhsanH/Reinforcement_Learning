import pygame
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

# --- COLOR & UI CUSTOM ---
ORANGE = (255, 165, 0)
GREEN_PIPE = (110, 215, 120)
DARK_GREEN = (80, 180, 90)
BROWN_FLOOR = (222, 184, 135)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (135, 206, 235) 
PIPE_HEAD_HEIGHT = 15 

# --- PYGAME INITIALIZATION ---
pygame.init()

GAME_WIDTH = 512
GAME_HEIGHT = 288
SCALE_FACTOR = 1.5 
SCREEN_WIDTH = int(GAME_WIDTH * SCALE_FACTOR)
SCREEN_HEIGHT = int(GAME_HEIGHT * SCALE_FACTOR)

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Box-Bird DQN")
GAME_SURFACE = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))

# Game Constants
GRAVITY = 0.25
BIRD_JUMP = -7
PIPE_SPEED = 3
PIPE_WIDTH = 52
PIPE_SPAWN_INTERVAL = 90  # Frame-based, bukan time-based
PIPE_GAP = 100
FLOOR_HEIGHT = 40

def draw_text_shadow(surface, text, x, y, color, shadow_color):
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, shadow_color)
    surface.blit(text_surface, (x + 1, y + 1))
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))

# --- GAME OBJECTS ---

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([20, 20], pygame.SRCALPHA)
        pygame.draw.rect(self.image, ORANGE, (0, 0, 20, 20), border_radius=3)
        pygame.draw.circle(self.image, BLACK, (15, 5), 3) 
        self.rect = self.image.get_rect(center=(GAME_WIDTH // 4, GAME_HEIGHT // 2))
        self.velocity = 0
        self.score = 0
        self.passed_pipe_x = set()

    def jump(self):
        self.velocity = BIRD_JUMP

    def update(self):
        self.velocity += GRAVITY
        self.rect.y += self.velocity
        # Pembatasan top
        if self.rect.top < 0:
            self.rect.top = 0

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x_pos, top_height, pipe_id):
        super().__init__()
        self.pipe_id = pipe_id
        self.is_top = True
        
        # Top Pipe
        self.image = pygame.Surface([PIPE_WIDTH, top_height], pygame.SRCALPHA)
        self.image.fill(GREEN_PIPE)
        pygame.draw.rect(self.image, DARK_GREEN, (0, max(0, top_height - PIPE_HEAD_HEIGHT), PIPE_WIDTH, PIPE_HEAD_HEIGHT))
        self.rect = self.image.get_rect(bottomleft=(x_pos, top_height))
        
        # Store bottom gap info
        self.gap_top = top_height
        self.gap_bottom = top_height + PIPE_GAP

    def update(self):
        self.rect.x -= PIPE_SPEED

class BottomPipe(pygame.sprite.Sprite):
    def __init__(self, x_pos, gap_bottom, pipe_id):
        super().__init__()
        self.pipe_id = pipe_id
        self.is_top = False
        
        bottom_height = GAME_HEIGHT - FLOOR_HEIGHT - gap_bottom
        
        # Bottom Pipe
        self.image = pygame.Surface([PIPE_WIDTH, bottom_height], pygame.SRCALPHA)
        self.image.fill(GREEN_PIPE)
        pygame.draw.rect(self.image, DARK_GREEN, (0, 0, PIPE_WIDTH, PIPE_HEAD_HEIGHT))
        self.rect = self.image.get_rect(topleft=(x_pos, gap_bottom))

    def update(self):
        self.rect.x -= PIPE_SPEED

# --- PIPE GENERATION ---
pipe_id_counter = 0

def create_pipe_pair(x_pos):
    global pipe_id_counter
    
    playable_height = GAME_HEIGHT - FLOOR_HEIGHT
    min_top_height = 40
    max_top_height = playable_height - PIPE_GAP - 40
    
    if max_top_height < min_top_height:
        top_height = playable_height // 2
    else:
        top_height = random.randint(min_top_height, max_top_height)
    
    top_height = max(10, min(top_height, playable_height - PIPE_GAP - 10))
    
    curr_id = pipe_id_counter
    pipe_id_counter += 1
    
    top_pipe = Pipe(x_pos, top_height, curr_id)
    bottom_pipe = BottomPipe(x_pos, top_height + PIPE_GAP, curr_id)
    
    return top_pipe, bottom_pipe

# --- DQN IMPLEMENTATION ---

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995 
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.memory = deque(maxlen=100000)

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_state(self, bird_obj, pipes_group):
        bird_y = bird_obj.rect.centery
        bird_velocity = bird_obj.velocity

        # Cari pipa berikutnya
        next_pipe = None
        min_dist = float('inf')
        
        for pipe in pipes_group:
            if isinstance(pipe, Pipe):  # Hanya top pipe
                if pipe.rect.right > bird_obj.rect.left:
                    dist = pipe.rect.left - bird_obj.rect.right
                    if dist < min_dist:
                        min_dist = dist
                        next_pipe = pipe
        
        if next_pipe is not None:
            gap_top = next_pipe.gap_top
            gap_bottom = next_pipe.gap_bottom
            pipe_x = next_pipe.rect.left
        else:
            gap_top = GAME_HEIGHT
            gap_bottom = GAME_HEIGHT
            pipe_x = GAME_WIDTH
        
        delta_y_top = gap_top - bird_y
        delta_y_bottom = gap_bottom - bird_y
        pipe_dist = pipe_x - bird_obj.rect.centerx
        
        state = [
            bird_y / GAME_HEIGHT,
            bird_velocity / 10.0,
            pipe_dist / GAME_WIDTH, 
            delta_y_top / GAME_HEIGHT,
            delta_y_bottom / GAME_HEIGHT,
        ]
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# --- GAME RESET ---

def reset_game():
    global bird, all_sprites, pipes, spawn_counter, pipe_id_counter
    
    all_sprites.empty()
    pipes.empty()
    spawn_counter = 0
    pipe_id_counter = 0
    
    bird = Bird()
    all_sprites.add(bird)
    
    # Spawn pipa pertama
    top_pipe, bottom_pipe = create_pipe_pair(GAME_WIDTH + 50)
    all_sprites.add(top_pipe, bottom_pipe)
    pipes.add(top_pipe, bottom_pipe)
    
    return bird

# --- MAIN TRAINING LOOP ---

all_sprites = pygame.sprite.Group()
pipes = pygame.sprite.Group()
spawn_counter = 0
pipe_id_counter = 0

STATE_DIM = 5
ACTION_DIM = 2

agent = DQNAgent(STATE_DIM, ACTION_DIM)
TRAIN_EPISODES = 10000
TARGET_UPDATE_FREQ = 50
VISUALIZE_INTERVAL = 1  # Tampilkan setiap episode 

clock = pygame.time.Clock()

for episode in range(TRAIN_EPISODES):
    bird = reset_game()
    current_state = agent.get_state(bird, pipes)
    
    is_game_over = False
    episode_reward = 0
    spawn_counter = 0

    while not is_game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Agent memilih aksi
        action = agent.choose_action(current_state)

        # Lakukan aksi
        if action == 1:
            bird.jump()

        # Update game
        all_sprites.update()
        
        # Spawn pipa berdasarkan frame
        spawn_counter += 1
        if spawn_counter >= PIPE_SPAWN_INTERVAL:
            top_pipe, bottom_pipe = create_pipe_pair(GAME_WIDTH)
            all_sprites.add(top_pipe, bottom_pipe)
            pipes.add(top_pipe, bottom_pipe)
            spawn_counter = 0
        
        # Hapus pipa yang sudah keluar
        for pipe in list(pipes):
            if pipe.rect.right < 0:
                pipes.remove(pipe)
                all_sprites.remove(pipe)
        
        # Hitung reward
        reward = 0.05
        
        # Collision check
        if bird.rect.bottom >= GAME_HEIGHT - FLOOR_HEIGHT or bird.rect.top <= 0:
            reward = -10
            is_game_over = True
        elif pygame.sprite.spritecollideany(bird, pipes):
            reward = -10
            is_game_over = True
        
        # Score untuk melewati tengah pipa (setengah jalan)
        for pipe in pipes:
            if isinstance(pipe, Pipe):
                pipe_center_x = pipe.rect.centerx
                if pipe.pipe_id not in bird.passed_pipe_x:
                    # Trigger saat bird melewati tengah pipa
                    if pipe_center_x - 5 < bird.rect.centerx < pipe_center_x + 5:
                        bird.passed_pipe_x.add(pipe.pipe_id)
                        bird.score += 1
                        reward += 5

        episode_reward += reward

        # Get next state
        next_state = agent.get_state(bird, pipes)

        # Learn
        agent.remember(current_state, action, reward, next_state, is_game_over)
        agent.learn()

        current_state = next_state

        # Visualisasi
        if episode % VISUALIZE_INTERVAL == 0:
            GAME_SURFACE.fill(LIGHT_BLUE)
            
            # Floor
            pygame.draw.rect(GAME_SURFACE, BROWN_FLOOR, (0, GAME_HEIGHT - FLOOR_HEIGHT, GAME_WIDTH, FLOOR_HEIGHT))
            for i in range(0, GAME_WIDTH, 20):
                pygame.draw.line(GAME_SURFACE, BLACK, (i, GAME_HEIGHT - FLOOR_HEIGHT), (i, GAME_HEIGHT), 1)

            all_sprites.draw(GAME_SURFACE)

            # UI Text
            draw_text_shadow(GAME_SURFACE, f"Score: {bird.score}", 10, 10, YELLOW, BLACK)
            draw_text_shadow(GAME_SURFACE, f"Episode: {episode+1}/{TRAIN_EPISODES}", 10, 40, WHITE, BLACK)
            draw_text_shadow(GAME_SURFACE, f"Epsilon: {agent.epsilon:.4f}", 10, 70, WHITE, BLACK)
            
            scaled_surface = pygame.transform.scale(GAME_SURFACE, (SCREEN_WIDTH, SCREEN_HEIGHT))
            SCREEN.blit(scaled_surface, (0, 0))
            
            pygame.display.flip()
            clock.tick(60) 
        
        else:
            pygame.event.pump()
            clock.tick(60)

    # Episode done
    print(f"Episode {episode+1}: Score = {bird.score}, Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

    # Update target network
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

pygame.quit()
sys.exit()
