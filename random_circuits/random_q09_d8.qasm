OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7],q[3];
cx q[8],q[5];
rz(5.8830014) q[6];
cx q[1],q[2];
rz(4.7003378) q[4];
rz(3.2741741) q[0];
rz(6.0243312) q[4];
cx q[5],q[3];
rz(0.37998537) q[7];
cx q[0],q[8];
rz(5.9589409) q[6];
rz(0.87606138) q[2];
rz(5.8336057) q[1];
rz(5.6605301) q[2];
cx q[1],q[8];
rz(1.8969909) q[0];
rz(2.4872902) q[6];
rz(5.7133562) q[7];
cx q[5],q[3];
rz(4.7236346) q[4];
rz(2.5758362) q[1];
rz(0.58716464) q[3];
rz(1.168553) q[2];
rz(5.3661541) q[7];
rz(1.0200828) q[8];
cx q[0],q[4];
cx q[6],q[5];
cx q[5],q[1];
rz(6.0078541) q[0];
rz(4.2678691) q[2];
cx q[3],q[6];
cx q[7],q[4];
rz(1.1647922) q[8];
cx q[1],q[0];
cx q[3],q[7];
cx q[5],q[2];
cx q[8],q[6];
rz(1.3458571) q[4];
cx q[2],q[0];
rz(0.14576657) q[1];
cx q[7],q[5];
cx q[3],q[8];
rz(4.4068251) q[6];
rz(1.4662212) q[4];
rz(3.1477033) q[3];
rz(0.22045112) q[7];
rz(1.7006803) q[6];
rz(4.5154186) q[4];
rz(4.2725869) q[8];
rz(4.7506016) q[2];
rz(0.82682819) q[0];
rz(2.8240208) q[5];
rz(2.8024681) q[1];
