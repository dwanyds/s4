#Setup smtp
import smtplib
def sendMailtrans(to_email,url,body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "colife0007@gmail.com"
    smtp_password = "xocxcfnbqoxolodr"

    smtp_conn = smtplib.SMTP(smtp_server, smtp_port)

    smtp_conn.ehlo()
    smtp_conn.starttls()
    smtp_conn.login(smtp_username, smtp_password)

    from_email = smtp_username
    # Sending email with text transcript
    subject = "Text Summary of "+url


    msg = f"From: {from_email}\nTo: {to_email}\nSubject: {subject}\n\n{body}"

    smtp_conn.sendmail(from_email, to_email, msg)