#!/bin/bash
host=$(hostname)
sudo echo "127.0.0.1 $host" >> /etc/hosts

sudo chown -R ubuntu /home/ubuntu
sudo chgrp -R ubuntu /home/ubuntu

mysql -uroot -proot -e "CREATE USER 'wordpressuser'@'%' IDENTIFIED BY 'wordpress'"
mysql -uroot -proot -e "GRANT ALL PRIVILEGES ON *.* TO 'wordpressuser'@'%' WITH GRANT OPTION"
mysql -uroot -proot -e "FLUSH PRIVILEGES"
mysql -uroot -proot -e "CREATE DATABASE wordpress"

#Make sure to replace the IP in the WP DB schema to the WEB SRV LB before you dumb it
#This is a smple regex that doesn't work in all cases of IP formats
sed -r -i "s/(\b[0-9]{1,3}\.){3}[0-9]{1,3}/129.115.160.16/g" /home/ubuntu/codesnippets/wp_db.sql
#Dump database `wordpress` instead of installing it manually using GUI
mysql -uroot -proot wordpress < /home/ubuntu/codesnippets/wp_db.sql

#sudo mount -t cifs //$shared_fpath /home/ubuntu/mnt -o guest

#Change auditd log to the shared path
#vmid=$(cat /etc/machine-id)
#sudo sed -i "s/^log_file = .*/log_file = \/home\/ubuntu\/mnt\/$vmid.log/1" /etc/audit/auditd.conf
#sudo sed -i "s/^log_group = .*/log_group = ubuntu/1" /etc/audit/auditd.conf
#sudo service auditd restart


sudo python /home/ubuntu/codesnippets/collecting_agent/collecting_agent.py &
