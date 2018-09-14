# A predict model for the car license price

## Background
Lets look at one of games, there are numbers of person take 
part in biding for guessing lowest price. As limiting number 
of car license, only the highest n person can win the game. 
Specifically, everyone can bid three time which two time for 
showing average price and the last time for real.

## Base model
There are two base models.

1. Assume that the distribution of bid price in one game belongs
to Beta distribution family. The mission target is to predict
the fractile value of the distribution with the attendance 
number predicted and two average price real-time.

2. Assume that the attendance number related to the average 
price and the lowest price in last game, besides itself.
