import pygame as pg
import random
import os
import math
import neat
import matplotlib.pyplot as plt
import numpy as np

vec = pg.math.Vector2

# colours
black = (0, 0, 0)
red   = (255, 0, 0)

win_nets     = {}
highscore    = 0
genHighscore = 0
gen          = 0
rows         = 20 # width of each cube
width        = 400 # as well as height
total_width  = 1200
total_height = 800
pg.init()
pg.display.set_caption("Snake AI")

STAT_FONT = pg.font.SysFont("comicsans", 20)

class Cube():
    rows = rows
    w = width
    def __init__(self,start,dirnx=1,dirny=0,color=(82,82,82)):
        self.pos = start
        self.dirnx = dirnx
        self.dirny = dirny
        self.color = color
       
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)
 
    def draw(self, surface):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
 
        pg.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))

class Snake():
    rows = rows
    w = width
    def __init__(self, pos):
        self.dirnx = 1
        self.dirny = 0
        self.head = Cube((pos[0], pos[1]), self.dirnx, self.dirny)
        self.body = []
        self.turns = {}
        self.body.append(self.head)
 
    def move(self, direction):
        if direction.x == 1 and self.dirnx != -1 and self.dirnx != direction.x:
            self.dirnx = direction.x
            self.dirny = direction.y
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif direction.x == -1 and self.dirnx != 1 and self.dirnx != direction.x:
            self.dirnx = direction.x
            self.dirny = direction.y
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif direction.y == -1 and self.dirny != 1 and self.dirny != direction.y:
            self.dirnx = direction.x
            self.dirny = direction.y
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif direction.y == 1 and self.dirny != -1 and self.dirny != direction.y:
            self.dirnx = direction.x
            self.dirny = direction.y
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
 
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0],turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx,c.dirny)
                    
    def reset(self, pos):
        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
 
    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny
 
        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0]-1,tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0]+1,tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0],tail.pos[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0],tail.pos[1]+1)))
 
        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
       
    def draw(self, surface):
        for i, c in enumerate(self.body):
                c.draw(surface)

def randomSnack(rows, snake): 
    positions = snake.body
 
    while True:
        x = random.randrange(0, rows-1)
        y = random.randrange(0, rows-1)
        if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
            continue
        else:
            break
       
    return (x,y)


def drawGrid(win):
    global total_height, total_width, width, rows
    pg.draw.line(win, black, (0,int(total_height/2)),(total_width,int(total_height/2)),2)
    for i in range(total_width//width):
        pg.draw.line(win, black, (int(width*i),0),(int(width*i),total_height),2)


def update_win(win, snakes, snacks, gen, scores, replay, gens = None):
    global rows, highscore, genHighscore
    win.fill((103,155,0))
    for snake in snakes:
        snake.draw(win)
    for snack in snacks:
        snack.draw(win)
    # drawGrid(win)

    
    for snake in snakes:
    # score
        score_label = STAT_FONT.render("Score: " + str(scores[snakes.index(snake)]), 1, red)
        win.blit(score_label, ((width - 100), 10))

        if replay:
            # expected score
            score_label = STAT_FONT.render("Expected: " + str(gens[snake][1]), 1, red)
            win.blit(score_label, ((width-130), 40))
    if not replay:
        # generations
        score_label = STAT_FONT.render("Gene: " + str(gen), 1, red)
        win.blit(score_label, (10, 10))
        # Generation with highscore
        score_label = STAT_FONT.render("At Gene: " + str(genHighscore), 1, red)
        win.blit(score_label, (10, 40))  
        # highscore
        score_label = STAT_FONT.render("Highscore: " + str(highscore), 1, red)
        win.blit(score_label, (10, width - 40))
    if replay:
        # generations
        score_label = STAT_FONT.render("Gene: " + str(gens[snake][0]), 1, red)
        win.blit(score_label, (10, 10))
        # Generation with highscore
        score_label = STAT_FONT.render("At Gene: " + str(genHighscore), 1, red)
        win.blit(score_label, (10, 40))  
        # highscore
        score_label = STAT_FONT.render("Highscore: " + str(highscore), 1, red)
        win.blit(score_label, (10, width - 40))
  
    pg.display.update()

def update_win_testwinners(win, snakes, snacks, scores):
    global rows, highscore
    win.fill((103,155,0))
    for snake in snakes:
        snake.draw(win)
    for snack in snacks:
        snack.draw(win)
    # drawGrid(win)

    
    for snake in snakes:
    # score
        score_label = STAT_FONT.render("Score: " + str(scores[snakes.index(snake)]), 1, red)
        win.blit(score_label, ((width-100), 10))

    # highscore
    score_label = STAT_FONT.render("Highscore: " + str(highscore), 1, red)
    win.blit(score_label, (10, 10))

    pg.display.update()
    
class CustomReporter(neat.reporting.BaseReporter):
    def __init__(self, stats):
        self.stats = stats

    def post_evaluate(self, config, population, species, best_genome):
        generation = len(self.stats.generation)
        self.stats.generation.append(generation)
        fitnesses = [c.fitness for c in population.values()]
        self.stats.max_fitness.append(max(fitnesses))
        self.stats.avg_fitness.append(sum(fitnesses) / len(fitnesses))
        self.stats.min_fitness.append(min(fitnesses))

class NEATStatistics:
    def __init__(self):
        self.generation = []
        self.max_fitness = []
        self.avg_fitness = []
        self.min_fitness = []



def plot_performance(stats):
    generations = stats.generation
    max_fitness = stats.max_fitness
    avg_fitness = stats.avg_fitness
    min_fitness = stats.min_fitness
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, label='Max Fitness')
    plt.plot(generations, avg_fitness, label='Avg Fitness')
    plt.plot(generations, min_fitness, label='Min Fitness')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('NEAT Algorithm Performance')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))
    
    # Add custom stats recording
    stats_1 = NEATStatistics()
    custom_reporter = CustomReporter(stats_1)
    p.add_reporter(custom_reporter)

    # Run for up to x generations.
    winner = p.run(run_game, 100)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    

    #Run the record breakers
    run_winners()

    #Test the best net x numbers of times 
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    test_winner(winner_net,120)
    
    # Plot the performance
    plot_performance(stats_1)

def run_game(genomes, config):
    global rows, width, total_height,total_width, gen, highscore, genHighscore, win_nets
    clock = pg.time.Clock()
    ge = []
    nets = []
    snakes = []
    snacks = []
    frames = []
    scores = []
    showGame = False
    max_frames = int(rows*rows/2)
    win = pg.display.set_mode((width, width))

    #print("no of genomes:"+ str(len(genomes)))

    for genome_id, genome in genomes:
        showGame = False
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snake = Snake((5,5))
        snakes.append(snake)
        snacks.append(Cube(randomSnack(rows, snake)))
        ge.append(genome)
        frames.append(0)
        scores.append(genome.fitness)
    
        if True:
            while len(snakes) > 0:
                if showGame:
                    #pg.time.delay(10) #Turn this on if you want the game to go slower
                    #clock.tick(10) 
                    update_win(win,snakes,snacks,gen,scores,False)
                    
                
                for index, snake in enumerate(snakes):
                    output = nets[index].activate(vision(snake,snacks[index]))

                    snake.move(getDirAction(snake, output))
                    if snake.body[0].pos == snacks[index].pos:
                        #print("YUM")
                        snake.addCube()
                        ge[index].fitness += 1
                        scores[index] = ge[index].fitness
                        snacks[index] = Cube(randomSnack(rows, snake))
                        frames[index] = 0
                        if ge[index].fitness > highscore:
                            showGame = True
                            showGame = False #to disable display. It may not work even if enabled.

                    frames[index] += 1
                    if frames[index] >= 100 and len(snake.body) <= 5:
                        frames[index] = max_frames
                        ge[index].fitness -= 10

                    for x in range(len(snake.body)):
                        if snake.body[x].pos in list(map(lambda z:z.pos,snake.body[x+1:])) or frames[index] >= max_frames:
                            frames.pop(index)
                            snacks.pop(index)
                            nets.pop(index)
                            ge.pop(index)
                            scores.pop(index)
                            snakes.remove(snake)
                            break
                        elif snake.head.pos[0] < 0 or snake.head.pos[0] > (snake.rows - 1) or snake.head.pos[1] > (snake.rows - 1) or snake.head.pos[1] < 0:
                            frames.pop(index)
                            snacks.pop(index)
                            nets.pop(index)
                            ge.pop(index)
                            scores.pop(index)
                            snakes.remove(snake)
                            break
                        
                if len(scores) > 0:
                    if max(scores) > highscore:
                        highscore = max(scores)
                        genHighscore = gen
                        win_nets[nets[snakes.index(snake)]] = [gen,highscore]
    
    gen += 1
    
def run_winners():
    global rows, width, total_height,total_width, gen, highscore, genHighscore, win_nets
    highscore = 0
    genHighscore = 0

    clock = pg.time.Clock()
    snakes = []
    snacks = []
    frames = []
    scores = []
    nets = []
    gens = {}
    max_frames = int(rows*rows/2)
    win = pg.display.set_mode((width, width))

    for net in win_nets:
        snake = Snake((5,5))
        snakes.append(snake)
        snacks.append(Cube(randomSnack(rows, snake)))
        frames.append(0)
        scores.append(0)
        nets.append(net)
        gens[snake] = win_nets[net]

        #if len(snakes) == 6 or len(gens) == len(win_nets):
        if True:
            while len(snakes) > 0:
                #pg.time.delay(10) 
                #clock.tick(10) #Turn this on if you want the game to go slower
                update_win(win,snakes,snacks,gen,scores,True,gens)
                
                for index, snake in enumerate(snakes):
                    output = nets[index].activate(vision(snake,snacks[index]))

                    snake.move(getDirAction(snake, output))
                    if snake.body[0].pos == snacks[index].pos:
                        snake.addCube()
                        scores[index] += 1
                        snacks[index] = Cube(randomSnack(rows, snake))
                        frames[index] = 0
                    
                    frames[index] += 1
                    if frames[index] >= 100 and len(snake.body) <= 5:
                        frames[index] = max_frames

                    for x in range(len(snake.body)):
                        if snake.body[x].pos in list(map(lambda z:z.pos,snake.body[x+1:])) or frames[index] >= max_frames:
                            frames.pop(index)
                            snacks.pop(index)
                            nets.pop(index)
                            scores.pop(index)
                            snakes.remove(snake)
                            break
                        elif snake.head.pos[0] < 0 or snake.head.pos[0] > (snake.rows - 1) or snake.head.pos[1] > (snake.rows - 1) or snake.head.pos[1] < 0:
                            frames.pop(index)
                            snacks.pop(index)
                            nets.pop(index)
                            scores.pop(index)
                            snakes.remove(snake)
                            break
                        
                if len(scores) > 0:
                    if max(scores) > highscore:
                        highscore = max(scores)
                        generation = gens[snake]
                        generation = generation[0]
                        genHighscore = generation


def test_winner(winner, n):
    global rows, width, total_height,total_width, gen, highscore, genHighscore, win_nets
    highscore = 0
    genHighscore = 0

    clock = pg.time.Clock()
    snakes = []
    snacks = []
    frames = []
    scores = []
    nets = []
    max_frames = int(rows*rows/2)
    win = pg.display.set_mode((width, width))

    for _ in range(n):
        snake = Snake((5,5))
        snakes.append(snake)
        snacks.append(Cube(randomSnack(rows, snake)))
        frames.append(0)
        scores.append(0)
        nets.append(winner)

        if True:
            while len(snakes) > 0:
                #pg.time.delay(10) #Turn this on if you want the game to go slower
                #clock.tick(10)
                update_win_testwinners(win,snakes,snacks,scores)

                for index, snake in enumerate(snakes):
                    output = nets[index].activate(vision(snake,snacks[index]))

                    snake.move(getDirAction(snake, output))
                    if snake.body[0].pos == snacks[index].pos:
                        snake.addCube()
                        scores[index] += 1
                        snacks[index] = Cube(randomSnack(rows, snake))
                        frames[index] = 0
                    
                    frames[index] += 1
                    if frames[index] >= 100 and len(snake.body) <= 5:
                        frames[index] = max_frames

                    for x in range(len(snake.body)):
                        if snake.body[x].pos in list(map(lambda z:z.pos,snake.body[x+1:])) or frames[index] >= max_frames:
                            frames.pop(index)
                            snacks.pop(index)
                            nets.pop(index)
                            scores.pop(index)
                            snakes.remove(snake)
                            break
                        elif snake.head.pos[0] < 0 or snake.head.pos[0] > (snake.rows - 1) or snake.head.pos[1] > (snake.rows - 1) or snake.head.pos[1] < 0:
                            frames.pop(index)
                            snacks.pop(index)
                            nets.pop(index)
                            scores.pop(index)
                            snakes.remove(snake)
                            break
                        
                if len(scores) > 0:
                    if max(scores) > highscore:
                        highscore = max(scores)



def vision(snake, snack):
    global rows
    dist = [-1,-1,-1] #AHEAD,LEFT,RIGHT
    distBody = [-1,-1,-1] #If body if 1 away AHEAD, LEFT, RIGHT
    defaultDist = rows/2

    head_x, head_y = snake.head.pos
    for i, body in enumerate(snake.body[1:]):
 
        #GOING RIGHT
        if snake.dirnx == 1:
            if (head_x + defaultDist) >= body.pos[0] and head_y == body.pos[1] and head_x < body.pos[0]: #BODY FORWARD
                if dist[0] == -1 or dist[0] > abs(head_x - body.pos[0]):
                    dist[0] = abs(head_x - body.pos[0])
                    if dist[0] == 1:
                        distBody[0] = 1
            if head_x == body.pos[0] and (head_y - defaultDist) <= body.pos[1] and head_y > body.pos[1]: #LEFT
                if dist[1] == -1 or dist[1] > abs(head_y - body.pos[1]):
                    dist[1] = abs(head_y - body.pos[1])
                    if dist[1] == 1:
                        distBody[1] = 1
            if head_x == body.pos[0] and (head_y + defaultDist) >= body.pos[1] and head_y < body.pos[1]: #RIGHT
                if dist[2] == -1 or dist[2] > abs(head_y - body.pos[1]):
                    dist[2] = abs(head_y - body.pos[1])
                    if dist[2] == 1:
                        distBody[2] = 1
        #GOING LEFT
        elif snake.dirnx == -1:
            if (head_x - defaultDist) <= body.pos[0] and head_y == body.pos[1] and head_x > body.pos[0]: #BODY FORWARD
                if dist[0] == -1 or dist[0] > abs(head_x - body.pos[0]):
                    dist[0] = abs(head_x - body.pos[0])
                    if dist[0] == 1:
                        distBody[0] = 1
            if head_x == body.pos[0] and (head_y + defaultDist) >= body.pos[1] and head_y < body.pos[1]: #LEFT
                if dist[1] == -1 or dist[1] > abs(head_y - body.pos[1]):
                    dist[1] = abs(head_y - body.pos[1])
                    if dist[1] == 1:
                        distBody[1] = 1
            if head_x == body.pos[0] and (head_y - defaultDist) <= body.pos[1] and head_y > body.pos[1]: #RIGHT
                if dist[2] == -1 or dist[2] > abs(head_y - body.pos[1]):
                    dist[2] = abs(head_y - body.pos[1])
                    if dist[2] == 1:
                        distBody[2] = 1
        #GOING UP
        elif snake.dirny == -1:
            if (head_y - defaultDist) <= body.pos[1] and head_x == body.pos[0] and head_y > body.pos[1]: #BODY FORWARD
                if dist[0] == -1 or dist[0] > abs(head_y - body.pos[1]):
                    dist[0] = abs(head_y - body.pos[1])
                    if dist[0] == 1:
                        distBody[0] = 1
            if head_y == body.pos[1] and (head_y - defaultDist) <= body.pos[0] and head_x > body.pos[0]: #LEFT
                if dist[1] == -1 or dist[1] > abs(head_x - body.pos[0]):
                    dist[1] = abs(head_x - body.pos[0])
                    if dist[1] == 1:
                        distBody[1] = 1
            if head_y == body.pos[1] and (head_x + defaultDist) >= body.pos[0] and head_x < body.pos[0]: #RIGHT
                if dist[2] == -1 or dist[2] > abs(head_x-body.pos[0]):
                    dist[2] = abs(head_x - body.pos[0])
                    if dist[2] == 1:
                        distBody[2] = 1                    

        #GOING DOWN 
        elif snake.dirny == 1:
            if (head_y + defaultDist) >= body.pos[1] and head_x == body.pos[0] and head_y < body.pos[1]: #BODY FORWARD
                if dist[0] == -1 or dist[0] > abs(head_y - body.pos[1]):    
                    dist[0] = abs(head_y - body.pos[1])
                    if dist[0] == 1:
                        distBody[0] = 1
            if head_y == body.pos[1] and (head_x + defaultDist) >= body.pos[0] and head_x < body.pos[0]: #LEFT
                if dist[1] == -1 or dist[1] > abs(head_x - body.pos[0]):
                    dist[1] = abs(head_x - body.pos[0])
                    if dist[1] == 1:
                        distBody[1] = 1
            if head_y == body.pos[1] and (head_x - defaultDist) <= body.pos[0] and head_x > body.pos[0]: #RIGHT
                if dist[2] == -1 or dist[2] > abs(head_x - body.pos[0]):
                    dist[2] = abs(head_x - body.pos[0])
                    if dist[2] == 1:
                        distBody[2] = 1

    #Adds vision of walls
    wallDist = distWall(snake)
    for i, wall in enumerate(wallDist):
        if wall != -1 and dist[i] == -1:
            dist[i] = wall


    #Getting for the direction of the snack
    dirSnack = [-1,-1,-1] #AHEAD, LEFT, RIGHT
    xDist = abs(head_x - snack.pos[0])
    yDist = abs(head_y - snack.pos[1])
    block = [-1,-1,-1] #BLOCKED BY BODY AHEAD, LEFT, RIGHT

    if snake.dirnx == 1:
        if head_x < snack.pos[0]:
            if dist[0] < xDist and dist[0] != -1:
                block[0] = 1
            else:
                dirSnack[0] = 1#abs(snake.head.pos[0]-snack.pos[0])
        elif head_x > snack.pos[0] and head_y == snack.pos[1]:
            if(random.randint(0,1)):
                dirSnack[1] = 1
            else:
                dirSnack[2] = 1
        if head_y > snack.pos[1]:
            if dist[1] < yDist and dist[1] != -1:
                block[1] = 1
            else:
                dirSnack[1] = 1#abs(snake.head.pos[1]-snack.pos[1])
        if head_y < snack.pos[1]:
            if dist[2] < yDist and dist[2] != -1:
                block[2] = 1
            else:
                dirSnack[2] = 1#abs(snake.head.pos[1]-snack.pos[1])

        
    elif snake.dirnx == -1:
        if head_x > snack.pos[0]:
            if dist[0] < xDist and dist[0] != -1:
                block[0] = 1
            else:
                dirSnack[0] = 1#abs(snake.head.pos[0]-snack.pos[0])
        elif head_x < snack.pos[0] and head_y == snack.pos[1]:
            if(random.randint(0,1)):
                dirSnack[1] = 1
            else:
                dirSnack[2] = 1
        if head_y < snack.pos[1]:
            if dist[1] < yDist and dist[1] != -1:
                block[1] = 1
            else:
                dirSnack[1] = 1#abs(snake.head.pos[1]-snack.pos[1])
        if head_y > snack.pos[1]:
            if dist[2] < yDist and dist[2] != -1:
                block[2] = 1
            else:
                dirSnack[2] = 1#abs(snake.head.pos[1]-snack.pos[1])

       
    elif snake.dirny == -1: 
        if head_y > snack.pos[1]:
            if dist[0] < yDist and dist[0] != -1:
                block[0] = 1
            else:
                dirSnack[0] = 1#abs(snake.head.pos[1]-snack.pos[1])
        elif head_y < snack.pos[1] and head_x == snack.pos[0]:
            if(random.randint(0,1)):
                dirSnack[1] = 1
            else:
                dirSnack[2] = 1
        if head_x > snack.pos[0]:
            if dist[1] < xDist and dist[1] != -1:
                block[1] = 1
            else:
                dirSnack[1] = 1#abs(snake.head.pos[0]-snack.pos[0])
        if head_x < snack.pos[0]:
            if dist[2] < xDist and dist[2] != -1:
                block[2] = 1
            else:
                dirSnack[2] = 1#abs(snake.head.pos[0]-snack.pos[0])


    elif snake.dirny == 1: 
        if head_y < snack.pos[1]:
            if dist[0] < yDist and dist[0] != -1:
                block[0] = 1
            else:
                dirSnack[0] = 1#abs(snake.head.pos[1]-snack.pos[1])
        elif head_y > snack.pos[1] and head_x == snack.pos[0]:
            if(random.randint(0,1)):
                dirSnack[1] = 1
            else:
                dirSnack[2] = 1
        if head_x < snack.pos[0]:
            if dist[1] < xDist and dist[1] != -1:
                block[1] = 1
            else:
                dirSnack[1] = 1#abs(snake.head.pos[0]-snack.pos[0])
        if head_x > snack.pos[0]:
            if dist[2] < xDist and dist[2] != -1:
                block[2] = 1
            else:
                dirSnack[2] = 1#abs(snake.head.pos[0]-snack.pos[0])
    
    if -1 not in dist:
        dirSnack = [-1,-1,-1]
        for i in range(len(dist)): 
            dirSnack[dist.index(max(dist))] = 1

    elif sum(block) > -2 or (1 in block and 1 in wallDist):
        dirSnack = [-1,-1,-1]
        dirSnack[dist.index(-1)] = 1             

    #print("V:"+str(dirSnack+dist))
    return dirSnack+dist


def distWall(snake):
    global rows
    defaultDist = 5
    dist = [-1,-1,-1] #AHEAD, LEFT, RIGHT
    
    head_x, head_y = snake.head.pos
    if snake.dirnx == 1:
        if (head_x + defaultDist) >= (rows-1):
            dist[0] = abs(head_x - (rows-1))
        if (head_y - defaultDist) <= 0:
            dist[1] = abs(snake.head.pos[1] - 0)
        if (head_y + defaultDist) >= (rows-1):
            dist[2] = abs(head_y - (rows-1))
    elif snake.dirnx == -1:  
        if (head_x - defaultDist) <= 0:
            dist[0] = abs(head_x)
        if (head_y - defaultDist) <= 0:
            dist[2] = abs(snake.head.pos[1] - 0)
        if (head_y + defaultDist) >= (rows-1):
            dist[1] = abs(head_y - (rows-1))
    elif snake.dirny == -1: 
        if (head_y - defaultDist) <= 0:
            dist[0] = abs(head_y - 0)
        if  (head_x + defaultDist) >= (rows - 1):
            dist[2] = abs(head_x - (rows - 1))
        if (head_x - defaultDist) <= 0:
            dist[1] = abs(head_x)
    elif snake.dirny == 1: 
        if (head_y + defaultDist) >= (rows - 1):
            dist[0] = abs(head_y - (rows-1))
        if (head_x + defaultDist) >= (rows-1):
            dist[1] = abs(head_x - (rows-1))
        if  (head_x - defaultDist) <= 0:
            dist[2] = abs(head_x)

    return dist

def getDirAction(snake, output):
    action = vec(0,0)
    #Calculating which direction is which depending on the current state
    if max(output) == output[0]: #LEFT
        if snake.dirnx == 1:
            action.x = 0
            action.y = -1
        elif snake.dirnx == -1:
            action.x = 0
            action.y = 1
        elif snake.dirny == -1:
            action.x = -1
            action.y = 0
        elif snake.dirny == 1:
            action.x = 1
            action.y = 0
    elif max(output) == output[1]: #RIGHT
        if snake.dirnx == 1:
            action.x = 0
            action.y = 1
        elif snake.dirnx == -1:
            action.x = 0
            action.y = -1
        elif snake.dirny == -1:
            action.x = 1
            action.y = 0
        elif snake.dirny == 1:
            action.x = -1
            action.y = 0
    elif max(output) == output[2]: #FORWARD:
        action.x = snake.dirnx
        action.y = snake.dirny

    return action



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
