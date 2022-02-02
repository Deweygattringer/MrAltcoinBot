
#!/bin/bash

echo "###########"
date

# don't change anything above this line

# important: no spaces before or after equal sign 
SCRIPT_DIRECTORY="/home/pi/gateiobot"
SCRIPT_NAME="main.py"
WAIT_SECONDS=3

# don't change anything below this line


function start_script() {
    # save old working directory - only needed for manual testing
    # but doesn't matter for cron
    OLD_WD=$PWD

    # change working directory
    cd "$SCRIPT_DIRECTORY"

    # restart script
    echo "starting $SCRIPT_NAME in the background"
    python3 "$SCRIPT_NAME" &

    # return to old working directory
    cd "$PWD"
}

# kill script
SCRIPT_PID=$(pgrep -f "$SCRIPT_NAME")
if [ -n "${SCRIPT_PID}" ];
then
    echo "$SCRIPT_NAME running - killing it "
    pkill -f "$SCRIPT_NAME"
else
    echo "$SCRIPT_NAME not running"
    start_script
    exit;
fi


# wait
sleep $WAIT_SECONDS

SCRIPT_PID=$(pgrep -f "$SCRIPT_NAME")
if [ -n "${SCRIPT_PID}" ];
then
    echo "$SCRIPT_NAME still running - killing it harder"
    pkill --signal SIGKILL -f "$SCRIPT_NAME"
else
    echo "$SCRIPT_NAME sucessfully killed"
fi

start_script