OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.70501891) q[4];
rz(1.2584534) q[4];
rz(0.34425371) q[4];
rz(2.2652586) q[0];
rz(2.257529) q[0];
rz(2.2364591) q[0];
rz(3.6859329) q[0];
rz(2.9296931) q[0];
rz(3.3886353) q[0];
rz(3.1699008) q[0];
rz(3.2412697) q[1];
rz(6.0589666) q[1];
rz(0.75780795) q[1];
rz(3.7626378) q[1];
rz(5.1799558) q[1];
rz(4.8972957) q[4];
rz(3.5459713) q[4];
rz(2.0504255) q[4];
rz(6.1699084) q[4];
rz(0.57138315) q[4];
rz(0.11825459) q[1];
rz(5.518644) q[4];
rz(1.0271397) q[4];
rz(5.4888672) q[7];
rz(2.7655492) q[1];
cx q[4],q[1];
rz(5.3208896) q[1];
cx q[0],q[1];
cx q[4],q[1];
cx q[1],q[0];
cx q[7],q[4];
rz(2.5508683) q[4];
rz(3.1717767) q[1];
cx q[4],q[1];
cx q[1],q[0];
cx q[4],q[7];
rz(2.2196909) q[0];
rz(1.5074236) q[0];
rz(0.92151118) q[0];
rz(4.8327784) q[0];
rz(1.0484009) q[0];
rz(3.025152) q[0];
rz(5.8010599) q[4];
rz(4.7249946) q[0];
rz(0.73134221) q[0];
rz(0.85777895) q[0];
cx q[7],q[4];
cx q[1],q[4];
rz(4.4724783) q[1];
rz(3.7537411) q[1];
