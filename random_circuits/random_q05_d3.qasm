OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[2],q[1];
rz(1.2710383) q[4];
rz(1.9704841) q[0];
rz(3.4633923) q[3];
rz(4.0606725) q[1];
cx q[2],q[3];
rz(0.070447938) q[0];
rz(1.5170564) q[4];
rz(5.0596967) q[3];
rz(2.4477692) q[1];
rz(1.6614067) q[4];
rz(3.7667318) q[2];
rz(1.6953718) q[0];
