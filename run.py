import pyautogui
import time
import game_board


def run(ai=None, report=True):
    """
    Main function, runs one game of the chrome dino game

    :param ai: strategy to use
    :param report: If you want to see printed messages during run
    :return: Score of AI (num of seconds
    """

    # Allow time to refresh and user to enter into the game
    if report:
        print('Start in: ')
        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)
    else:
        time.sleep(5)

    # Find the region of interest, and click there
    board = game_board.GameBoard()
    if board.found_roi:
        board.go_to_roi()
    else:
        print("Can't find game... exiting")
        return 0

    # allow some time for game to load and press space to start
    pyautogui.press('space')
    time.sleep(1)  # be easy - there's a starting animation

    game_on = True
    score = 0

    prior_obstacles = board.empty_obstacles
    last_stuck_check = time.time()

    # start game timer
    game_start = time.time()
    while game_on:

        # Take screen shot and find obstacles and score
        game_img = board.get_game_img()
        obstacles = board.find_obstacles(game_img)
        score = time.time() - game_start

        # Sometimes it takes too long to load and never starts in that case gets stuck.
        # Restart the function if it is long enough into the game, seeing an empty board and it happened last loop
        if time.time() - last_stuck_check > 10:
            last_stuck_check = time.time()

            if prior_obstacles == obstacles:
                print('game looks stuck! restarting!')
                return run(ai=ai, report=report)

            prior_obstacles = obstacles

        # Call game AI with game state - Uncomment AI you want to test/train
        if ai.jump(obstacles, score) >= .5:
            pyautogui.press('space')
            time.sleep(.5)  # so we don't mash the space bar and ruin everything

        # If game over, log score, and restart game
        if board.game_over(game_img):
            score = time.time() - game_start
            game_on = False
            if report:
                print('game over score: {0:0.1f}'.format(score))
            # restart
            pyautogui.hotkey('ctrl', 'r')

    return score

