#!/bin/sh
host=$(hostname)
sudo echo "127.0.0.1 $host" >> /etc/hosts

sed -i "s/database_name_here/$db_name/g" /var/www/html/wp-config.php
sed -i "s/username_here/$db_user/g" /var/www/html/wp-config.php
sed -i "s/password_here/$db_pass/g" /var/www/html/wp-config.php
sed -i "s/localhost/$db_host/g" /var/www/html/wp-config.php

#Change auditd log to the shared path
#vmid=$(cat /etc/machine-id)
#sudo sed -i "s/^log_file = .*/log_file = \/home\/ubuntu\/mnt\/$vmid.log/1" /etc/audit/auditd.conf
#sudo sed -i "s/^log_group = .*/log_group = ubuntu/1" /etc/audit/auditd.conf
#sudo service auditd restart

sudo chown -R ubuntu /home/ubuntu
sudo chgrp -R ubuntu /home/ubuntu
sudo chown -R www-data /var/www/html/*
sudo chgrp -R www-data /var/www/html/*

#sudo mount -t cifs //$shared_fpath /home/ubuntu/mnt -o guest

sudo python /home/ubuntu/codesnippets/collecting_agent/collecting_agent.py &
