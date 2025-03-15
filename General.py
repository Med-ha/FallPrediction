import os
import datetime
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class General:
    
    COMPUTER = os.name  # Identifying OS
    
    @staticmethod
    def time_cell(tdat=None):
        if tdat is None:
            tdat = datetime.datetime.now()
        
        str_time = tdat.strftime("%d-%b-%Y %H:%M")
        year = tdat.strftime("%Y")
        mon = tdat.strftime("%b")
        day = tdat.strftime("%d")
        hourmin = tdat.strftime("%H.%M")
        num_mon = tdat.strftime("%m")
        
        time_cell = [year, mon, day, hourmin, num_mon]
        return (time_cell, tdat) if tdat else time_cell
    
    @staticmethod
    def add_time_to_file(file_name_finale):
        time_cell = General.time_cell()[0]
        
        if General.COMPUTER == 'posix':  # Mac/Linux
            file_path = '/Users/avianoah/Dropbox/Avia_Research/Gracefall/Results'
        elif General.COMPUTER == 'nt':  # Windows
            file_path = 'C:\\Users\\Owner\\Dropbox\\Avia_Research\\Gracefall'
        else:
            raise OSError("Unsupported OS")
        
        file_name = os.path.join(file_path, time_cell[0], time_cell[1], time_cell[2] + '.txt')
        return file_name
    
    @staticmethod
    def file_make(file_name, titles):
        if not os.path.isfile(file_name):
            file_path = os.path.dirname(file_name)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            
            with open(file_name, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=' ')
                writer.writerow(titles)
    
    @staticmethod
    def log_info(file_name, data):
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerow(data)
    
    @staticmethod
    def set_mail():
        mail = 'qil@mail.huji.ac.il'
        psswd = 'Zaq12wsx'  # Consider using environment variables for security
        host = 'smtp.gmail.com'
        port = 465
        
        msg = MIMEMultipart()
        msg['From'] = mail
        msg['To'] = mail
        msg['Subject'] = 'subject'
        msg.attach(MIMEText('test', 'plain'))
        
        try:
            server = smtplib.SMTP_SSL(host, port)
            server.login(mail, psswd)
            server.send_message(msg)
            server.quit()
            print("Email sent successfully")
        except Exception as e:
            print(f"Error sending email: {e}")
