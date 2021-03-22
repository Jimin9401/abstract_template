from slacker import Slacker
import time
key="xoxb-1756076074501-1744406067495-6bXDmJXfN7A3KZlvJl8xVpGf"

# slackerBot=Slacker(key)
# slackerBot.chat.post_message("#alarm-jimmy","Starting training")

class Bot():
    def __init__(self  ,chat_room="#alarm-jimmy"):
        self.bot=Slacker(key)
        self.chat_room = chat_room

    def post_message(self, message):
        self.bot.chat.post_message(self.chat_room,message+" @Jimin Hong")

slackBot=Bot()

def slackalarm(training_func):
    def wrapper(*args, **kargs):
        try:
            start = time.time()
            training_func(*args, **kargs)
            end = time.time()
            msg = end - start
        except Exception as e:

            msg = e.args

        slackBot.post_message(" Total computation time is {0} secs".format(str(msg)))

    return wrapper
