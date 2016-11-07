from itertools import cycle
import random
import sys
import operator

from collections import defaultdict
import copy
import pygame
import itertools
from pygame.locals import *

iters = 0
games = 0

FPS = 3000
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}
    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
        }

class Decider:
    def __init__(self):
        self.t = 0
        self.flag = False
        self.discount = .95
        self.trace = []
        self.nstate = None
        self.seen = defaultdict(lambda: 0)
        self.historicState = None
        self.historicAction = None
        self.actions = [None, pygame.event.Event(KEYDOWN, key = K_UP)]
        self.q = defaultdict(lambda: -39.2) #exp value of bad path
        self.q[(('crashed',),0)] = -10000
        self.q[(('crashed',),1)] = -10000
    def handleActions(self, prefeatures):
       def processGlobalState(state):
           if 'crashed' in state:
               return ('crashed',)
           i = 0
           if state['playerx'] - 40 >= state['upperPipes'][0]['x']:
               i = 1
           closePipeOut = state['upperPipes'][i]['x'] - state['playerx']
           currY = state['playery']
           currYVel = state['playerVelY']
           lowerPipe = int(state['lowerPipes'][i]['y']) #delta
           #threshold
           #if currY <= 0:
           #    currY = 0
           #if closePipeOut >= 200:
           #    closePipeOut = 199
           closePipeOut = int(closePipeOut/4)
           currY = int(currY/4)
           lowerPipe = int(lowerPipe/4)
           return (closePipeOut, currY, lowerPipe, int(currYVel))
       currState = processGlobalState(prefeatures)
       if self.flag:
           print currState
       states = {i : processGlobalState(simulate(prefeatures,[self.actions[i]], lol='y')) for i in range(0,len(self.actions))} 
       # select state biased towards better choices
       actionIndex = max(states.iterkeys(), key=(lambda key: self.q[(states[key],key)]))
       bias = 0.14 #1.0/(1 + 2 * min([self.seen[(currState, act)] for act in range(0,len(self.actions))]))
       global FPS
       if FPS == 30:
           print "At ",currState,":"
           for i in range(0,len(self.actions)):
               print i,"->",states[i]," (cost: ",self.q[(states[i],0)]," seen ",self.seen[(states[i],0)]," and ",self.q[(states[i],1)],"seen ",self.seen[(states[i],1)],")"
       elif random.random() < bias:
           actionIndex = random.choice([0, 1])
       #compute reward
       if currState[0] == 'crashed':
           reward = -10000 #dead
       else:
           reward = 1 #alive
       playerMidPos = prefeatures['playerx'] + IMAGES['player'][0].get_width() / 2
       for pipe in prefeatures['upperPipes']:
           pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
           if pipeMidPos - 20 <= playerMidPos < pipeMidPos + 4 - 20:
               reward = 4000 #good!
               print "nothresh paid out ",currState
               #self.flag = True
       #update
       tup = (currState, actionIndex)
       maxValue = max([self.q[(states[actionIndex],i)] for i in range(0,len(self.actions))])
       #if self.q[tup] >= 0 and reward + self.discount * maxValue < 0:
       #    print str(tup)+" is now "+str(reward + self.discount * maxValue)
       #if self.q[tup] < 100 and reward + self.discount * maxValue > 100:
       #    print str(tup)+" is now "+str(reward + self.discount * maxValue)
       self.q[tup] = self.q[tup] + bias*(reward + self.discount * maxValue - self.q[tup])
       if FPS == 30:
           print tup," revalued to ",self.q[tup]
       #print tup, "has value", self.q[tup]
       self.seen[tup] = self.seen[tup] + 1
       self.historicState = currState
       self.historicAction = actionIndex
       return self.actions[actionIndex];
def simulate(v, events, lol="lol"):
    playerx = v['playerx']
    playery = v['playery']
    playerVelY = v['playerVelY']
    playerFlapAcc = v['playerFlapAcc']
    playerIndex = v['playerIndex']
    basex = v['basex']
    playerMaxVelY = v['playerMaxVelY']
    playerAccY = v['playerAccY']
    upperPipes = copy.deepcopy(v['upperPipes'])
    lowerPipes = copy.deepcopy(v['lowerPipes'])
    baseShift = v['baseShift']
    pipeVelX = v['pipeVelX']
    playerFlapped = False

    for event in events:
        if event is not None and playery > -2 * IMAGES['player'][0].get_height():
            playerVelY = playerFlapAcc
            playerFlapped = True
        for playerIndex in [0,1,2]:
            crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
            if crashTest[0]:
                return {"crashed": True}
        
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = max(IMAGES['player'], key=lambda x: x.get_height()).get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        
        if playerx - 40 >= upperPipes[0]['x']:
            upperPipes[0]['x'] = upperPipes[1]['x']
            upperPipes[0]['y'] = upperPipes[1]['y']
    return {'playerAccY': playerAccY, 'upperPipes': upperPipes, 'lowerPipes': lowerPipes, 'playerx': playerx, 'playery': playery, 'playerVelY': playerVelY}

def mainGame(movementInfo):
    global games
    decider = Decider()
    while True:
        try:
            games = games + 1
            playIt(movementInfo,decider)
            decider.t = decider.t + 1
        except:
            print "Exception: ", sys.exc_info()[0]
def playIt(movementInfo,decider):
    global iters,games,FPS
    itercount = 0
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    while True:
        eventLog = None
        for event in pygame.event.get():
            if event.type == KEYDOWN and (event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
            if event.type == KEYDOWN and (event.key == K_SPACE): 
                if FPS == 3000:
                    FPS = 4
                elif FPS == 4:
                    FPS = 30
                else:
                    FPS = 3000
        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        
        if crashTest[0]:
            iters = iters + itercount
            if itercount > 140:
                print str(itercount) + "versus " + str(iters/games)
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
            }
        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if FPS != 3000:
                    SOUNDS['point'].play()

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = playerIndexGen.next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        if FPS != 3000 or (itercount + iters) % 1000 == 0:
            SCREEN.blit(IMAGES['background'], (0,0))
       
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
            showScore(score)
            SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))

            pygame.display.update()
        worldState = {'x': playerx, 'y': playery, 'upperPipes': upperPipes, 'lowerPipes': lowerPipes, 'playerVelY': playerVelY} 
        action = decider.handleActions(locals())
        if action is not None:
            pygame.event.post(action)
        itercount = itercount + 1
        FPSCLOCK.tick(FPS)

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]
    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
