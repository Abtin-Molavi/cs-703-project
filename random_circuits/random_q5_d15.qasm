OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(1.0859972) q[0];
cx q[3],q[4];
cx q[1],q[2];
cx q[2],q[0];
cx q[3],q[1];
rz(3.8679753) q[4];
rz(1.4294707) q[2];
cx q[0],q[4];
rz(0.24050604) q[1];
rz(0.79913338) q[3];
rz(2.5941277) q[4];
rz(3.766451) q[0];
rz(3.0676491) q[1];
cx q[3],q[2];
cx q[1],q[3];
cx q[4],q[2];
rz(2.3680269) q[0];
cx q[0],q[2];
rz(2.6353652) q[4];
cx q[3],q[1];
rz(2.3996522) q[2];
cx q[4],q[3];
cx q[1],q[0];
rz(3.2490337) q[0];
cx q[4],q[2];
cx q[1],q[3];
cx q[1],q[0];
rz(4.6467816) q[4];
cx q[2],q[3];
rz(1.3764604) q[1];
rz(5.6546706) q[0];
cx q[3],q[4];
rz(4.6414106) q[2];
cx q[0],q[4];
cx q[1],q[3];
rz(2.1141525) q[2];
rz(4.1031638) q[2];
cx q[1],q[4];
rz(2.343205) q[0];
rz(1.8509836) q[3];
cx q[2],q[3];
cx q[4],q[1];
rz(2.7708847) q[0];
rz(2.6171086) q[1];
rz(0.53574791) q[2];
cx q[0],q[3];
rz(3.7579202) q[4];
rz(5.0131543) q[1];
rz(5.0642907) q[0];
cx q[3],q[2];
rz(2.9113882) q[4];
