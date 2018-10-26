
#!/usr/bin/bash

while true
do
	echo "Running"
	kubectl get pods | grep "Running" | grep "job*" | awk '{print $1}' | while read a; do kubectl cp $a:log .; done
	sleep 5
	kubectl get pods | grep "Completed" | awk '{print $1}' | while read a; do kubectl delete pod $a; done
done