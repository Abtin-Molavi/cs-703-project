OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(1.9837864) q[1];
rz(5.6538764) q[1];
rz(5.9724164) q[2];
rz(0.54292076) q[2];
rz(3.3698059) q[2];
rz(3.8106785) q[4];
rz(4.3579853) q[4];
rz(2.4614937) q[4];
rz(1.4456583) q[3];
rz(3.4009637) q[3];
rz(2.7384514) q[1];
rz(2.177915) q[1];
cx q[4],q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
rz(1.683972) q[2];
cx q[3],q[2];
rz(2.3886407) q[4];
