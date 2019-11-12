#!/bin/bash
host=$(hostname)
sudo echo "127.0.0.1 $host" >> /etc/hosts

sudo sed -i "/<VirtualHost \*:80>/a ProxyPassMatch ^\/(.\*\\\.php(\/.*)?)$ fcgi:\/\/$lb_ip:9000\/var\/www\/html\/\$1" /etc/apache2/sites-enabled/000-default.conf
sudo service apache2 restart

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

