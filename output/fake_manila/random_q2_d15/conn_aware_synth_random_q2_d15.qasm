OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(3.3163176) q[0];
rz(1.1340636) q[0];
rz(6.0932626) q[1];
rz(5.167615) q[0];
rz(4.8140773) q[0];
rz(1.1237645) q[0];
cx q[1],q[0];
rz(5.6329003) q[0];
rz(2.9904006) q[0];
rz(5.534203) q[0];
rz(3.5372172) q[0];
cx q[0],q[1];
