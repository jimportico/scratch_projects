import os
import time 
import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

"""
Checks Amazon Prime Now and posts delivery times
"""

class AmazonPrimeNow(webdriver.Chrome):
    def __init__(self, browser):
        self.site = "https://primenow.amazon.com/home"
        if browser.lower() == "chrome":
            self.driver = webdriver.Chrome.__init__(self)
        else:
            raise ValueError("Only Chrome is installed at this time")

    def open_site(self):
        # Navigate to Amazon Prime Now
        self.get(self.site)
        # Enter Postal Code
        self.find_element_by_name("lsPostalCode").send_keys("10009")
        # sleep
        time.sleep(np.random.randint(3, 8))
        self.find_element_by_class_name("a-button-input").click()
        # sleep
        time.sleep(np.random.randint(5, 8))
        self.fullscreen_window()

    def sign_in(self, amazon_un, amazon_pw):
        # Navigate to Log-on screen
        self.find_element_by_css_selector(
            """div[class="show-for-large page_header_drop_menu_icon__root__19BcV"]"""
        ).click()
        # sleep
        time.sleep(np.random.randint(3, 6))
        email = self.find_element_by_name("email")
        password = self.find_element_by_name("password")
        email.send_keys(amazon_un)
        password.send_keys(amazon_pw)
        # sleep
        time.sleep(np.random.randint(7, 8))
        self.find_element_by_id("signInSubmit").click()

    def check_out(self):
        # Go-to cart
        self.find_element_by_css_selector(
            """
            [aria-label="Cart"]
        """
        ).click()
        # sleep
        time.sleep(np.random.randint(4, 8))
        # Proceed to check out
        self.find_element_by_id("a-autoid-1").click()

    def _strip_time(self, times, categories):
        """strip unwanted categories"""
        stripped_times = [x for x in times.splitlines() if x not in categories]
        return stripped_times

    def _enumerate_time(self, strip_times):
        enum_times = [(x, y) for x, y in enumerate(strip_times)]
        return enum_times

    def check_availability(self):
        # Delivery times
        # sleep
        time.sleep(np.random.randint(4, 8))
        try:
            window = self.find_element_by_id("two-hour-window")
            times = window.text
            categories = [
                "Collapse all 2-Hour windows",
                "Tomorrow",
                "See all 2-hour windows",
            ]

            stripped_times = self._strip_time(times, categories)
            enum_times = self._enumerate_time(stripped_times)

            time_slot = [x[1] for x in enum_times if x[0] % 2 == 0]
            availability = [x[1] for x in enum_times if x[0] % 2 != 0]
            df = pd.DataFrame({"times": time_slot, "avail": availability})
            return df
        except Exception as e:
            print("No times available.\n\n", e)

    def email_alert(self, message=None, recipients=None, avail_df=None):
        sender_email = "xxxxx@xxxx.com"
        all_recipients = []
        for i in recipients:
            all_recipients.append(i)
        recipients = ", ".join(all_recipients)
        msg = MIMEMultipart()
        msg["Subject"] = "Food Delivery Update"
        msg["From"] = f"Food Delivery Monitor <{sender_email}>"
        msg["To"] = recipients
        msg.attach(MIMEText(message, "plain"))
        msg.attach(MIMEText(avail_df.to_string(), "plain"))
        session = smtplib.SMTP("mail.xxxxx.com")
        session.sendmail(sender_email, recipients, msg.as_string())
        session.quit()


# Kick off script
# ---------------
amazon_un = "xxxx"
amazon_pw = "xxxx"

message = "Your food is ready"
recipients = ["xxxx", "xxxx"]

counter = 0
num_min = 30        # How often we'd like to check Amazon for available times
run_time_hrs = 5    # Total run time in hours

if __name__ == "__main__":
    amzn = AmazonPrimeNow(browser="Chrome")
    amzn.open_site()
    amzn.sign_in(amazon_un=amazon_un, amazon_pw=amazon_pw)
    amzn.check_out()
    
    while True:
        # Refresh page
        # ------------
        amzn.refresh()
        # Store delivery times to df
        # --------------------------
        df = amzn.check_availability()
        # Execute email trigger
        # ---------------------
        if df is not None:
            amzn.email_alert(message=message, recipients=recipients, avail_df=df)
        else:
            print("No times available - no email sent.")

        # Keeping Track & Ending Routine
        # ------------------------------
        print(f"Executing loop {counter}...")
        sleep_time = num_min * 60
        time.sleep(sleep_time)
        counter += 1
        if counter > run_time_hrs:
            break