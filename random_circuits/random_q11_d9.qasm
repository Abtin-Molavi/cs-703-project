OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
rz(4.7175197) q[2];
rz(5.8609182) q[1];
cx q[6],q[5];
rz(6.1595749) q[9];
cx q[4],q[3];
rz(5.2301922) q[8];
cx q[10],q[7];
rz(4.2551212) q[0];
rz(4.2483024) q[8];
cx q[10],q[5];
cx q[6],q[9];
cx q[2],q[3];
rz(3.720084) q[7];
rz(4.8397112) q[1];
cx q[4],q[0];
cx q[5],q[2];
cx q[6],q[9];
cx q[1],q[10];
cx q[3],q[0];
cx q[8],q[7];
rz(1.8437181) q[4];
rz(1.5820423) q[1];
rz(2.7733338) q[7];
cx q[9],q[10];
cx q[2],q[6];
rz(3.709979) q[4];
cx q[3],q[5];
rz(1.3713382) q[8];
rz(0.22161422) q[0];
cx q[5],q[3];
cx q[8],q[1];
rz(3.4495611) q[6];
cx q[7],q[4];
cx q[9],q[0];
cx q[10],q[2];
rz(4.7194696) q[1];
rz(4.7322395) q[9];
cx q[2],q[6];
cx q[4],q[5];
cx q[7],q[8];
cx q[3],q[0];
rz(2.4565565) q[10];
rz(6.2454191) q[1];
cx q[4],q[9];
rz(5.1588372) q[8];
cx q[6],q[3];
cx q[7],q[10];
cx q[0],q[5];
rz(0.77664425) q[2];
rz(1.1795121) q[9];
rz(2.6948294) q[4];
cx q[6],q[1];
cx q[3],q[10];
cx q[8],q[0];
cx q[5],q[2];
rz(4.1525051) q[7];
cx q[5],q[9];
rz(3.1566266) q[3];
rz(4.6147753) q[10];
cx q[6],q[0];
rz(2.6307842) q[1];
rz(3.3857409) q[8];
rz(5.6411433) q[2];
rz(0.50116897) q[7];
rz(2.4668435) q[4];
