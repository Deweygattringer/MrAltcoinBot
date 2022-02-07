# LET THIS SCRIPT RUN AS CRONJOB (CRONTAB -E): 0 0 1 * *

import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
y = datetime.datetime.now()
monthy = (y.strftime("%B" " " "%Y"))
My = str(monthy)
mail_content = ('Hello,\n \nThis is your monthly performance report for your trading bot for ') + My
x = datetime.datetime.now()
month = (x.strftime("%b" " " "%Y"))
M = str(month)

#The mail addresses and password
#won't work with 2FA...you can create an app-pw (16 digits with spaces) via: security.google.com/settings/security/apppasswords
sender_address = 'sender@gmail.com'
sender_pass = 'xxxx xxxx xxxx xxxx'
receiver_address = 'recipient@gmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'Bot Performance Report:' + M
#The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))
#the profitsfile is formated to be importet to excel for analytics
attach_file_name = 'profits.txt'
attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
payload = MIMEBase('application', 'octate-stream')
payload.set_payload((attach_file).read())
encoders.encode_base64(payload) #encode the attachment
#add payload header with filename
payload.add_header('Content-Disposition', f'attachment; filename = {attach_file_name}')
message.attach(payload)
#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()

print('Mail Sent')
# erase the content of profits.txt
file = open('profits.txt', 'r+')
file.truncate(0)
file.close
