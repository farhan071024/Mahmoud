#!/usr/bin/sh

#NOTE: You may want to sleep few seconds between each command because they depend that the previous command is done first

net=private_subnet

#Create and configure webservers' loadbalancer
neutron lbaas-loadbalancer-create --name webserver_lb $net
sleep 10s
neutron lbaas-listener-create --loadbalancer webserver_lb --protocol HTTP --protocol-port 80 --name webserver_listener1
sleep 10s
neutron lbaas-pool-create --lb-algorithm ROUND_ROBIN --listener webserver_listener1 --protocol HTTP --name webserver_pool1

sleep 10s

#Create and configure webservers' loadbalancer
neutron lbaas-loadbalancer-create --name appserver_lb $net
sleep 10s
neutron lbaas-listener-create --loadbalancer appserver_lb --protocol TCP --protocol-port 9000 --name appserver_listener1
sleep 10s
neutron lbaas-pool-create --lb-algorithm ROUND_ROBIN --listener appserver_listener1 --protocol TCP --name appserver_pool1
