OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.70501891) q[1];
rz(1.2584534) q[1];
rz(0.34425371) q[1];
rz(2.2652586) q[2];
rz(2.257529) q[2];
rz(2.2364591) q[2];
rz(3.6859329) q[2];
rz(2.9296931) q[2];
rz(3.3886353) q[2];
rz(3.1699008) q[2];
rz(3.2412697) q[3];
rz(6.0589666) q[3];
rz(0.75780795) q[3];
rz(3.7626378) q[3];
rz(5.1799558) q[3];
rz(4.8972957) q[1];
rz(3.5459713) q[1];
rz(2.0504255) q[1];
rz(6.1699084) q[1];
rz(0.57138315) q[1];
rz(0.11825459) q[3];
rz(5.518644) q[1];
rz(1.0271397) q[1];
rz(5.4888672) q[0];
rz(2.7655492) q[3];
cx q[1],q[0];
rz(2.5508683) q[0];
rz(5.8010599) q[0];
cx q[3],q[1];
rz(5.3208896) q[1];
cx q[2],q[1];
cx q[0],q[2];
cx q[2],q[3];
cx q[3],q[0];
rz(3.1717767) q[0];
cx q[1],q[0];
rz(2.2196909) q[2];
rz(1.5074236) q[2];
rz(0.92151118) q[2];
rz(4.8327784) q[2];
rz(1.0484009) q[2];
rz(3.025152) q[2];
rz(4.7249946) q[2];
rz(0.73134221) q[2];
rz(0.85777895) q[2];
rz(4.4724783) q[3];
rz(3.7537411) q[3];