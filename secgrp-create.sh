#!/bin/bash

#For file sharing using smb, ipv6 ingress rules required
openstack security group create webserver-sec-grp
openstack security group rule create --proto tcp --dst-port 22:22 webserver-sec-grp
openstack security group rule create --proto tcp --dst-port 80:80 webserver-sec-grp
openstack security group rule create --proto tcp --dst-port 443:443 webserver-sec-grp
openstack security group rule create --proto icmp webserver-sec-grp

openstack security group create appserver-sec-grp
openstack security group rule create --proto tcp --dst-port 22:22 appserver-sec-grp
openstack security group rule create --proto tcp --dst-port 9000:9000 appserver-sec-grp
openstack security group rule create --proto icmp appserver-sec-grp

openstack security group create dbserver-sec-grp
openstack security group rule create --proto tcp --dst-port 22:22 dbserver-sec-grp
openstack security group rule create --proto tcp --dst-port 3306:3306 dbserver-sec-grp
openstack security group rule create --proto icmp dbserver-sec-grp
