OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[0],q[4];
rz(1.8653888) q[2];
rz(1.9141513) q[1];
rz(4.2121356) q[3];
rz(5.3430172) q[5];
cx q[0],q[2];
cx q[1],q[5];
cx q[4],q[3];
cx q[5],q[0];
cx q[4],q[2];
rz(5.7345425) q[1];
rz(2.559993) q[3];
cx q[4],q[5];
rz(1.3217289) q[2];
cx q[3],q[0];
rz(3.3689926) q[1];
cx q[2],q[3];
rz(4.8608712) q[4];
rz(3.701832) q[0];
cx q[5],q[1];
cx q[4],q[3];
cx q[2],q[1];
rz(4.6191764) q[5];
rz(1.3851397) q[0];
rz(5.5408585) q[5];
rz(1.0888136) q[0];
rz(5.6232853) q[3];
rz(1.8539778) q[1];
rz(3.5837073) q[4];
rz(5.2476982) q[2];
rz(4.6282933) q[2];
cx q[4],q[0];
rz(0.01418038) q[1];
cx q[3],q[5];
cx q[1],q[4];
rz(0.63645215) q[5];
rz(4.6023314) q[2];
rz(5.8008685) q[0];
rz(2.6986123) q[3];
cx q[3],q[4];
cx q[0],q[2];
cx q[1],q[5];
cx q[1],q[3];
rz(3.2861261) q[2];
rz(1.2036243) q[5];
rz(1.5919549) q[4];
rz(2.4749171) q[0];
cx q[1],q[5];
cx q[0],q[2];
cx q[3],q[4];
rz(3.7403228) q[4];
cx q[0],q[1];
rz(2.1578708) q[2];
cx q[3],q[5];
