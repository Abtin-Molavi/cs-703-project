OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[1],q[4];
rz(2.3809821) q[0];
rz(3.6173111) q[3];
cx q[2],q[6];
rz(2.5429955) q[5];
rz(2.3958697) q[6];
cx q[4],q[3];
rz(4.8470529) q[2];
rz(3.4557011) q[0];
rz(2.4582993) q[5];
rz(5.4789154) q[1];
rz(0.66728877) q[3];
cx q[5],q[4];
cx q[0],q[6];
cx q[1],q[2];
rz(5.4890838) q[0];
rz(5.1897913) q[1];
rz(2.3551688) q[6];
rz(1.3367934) q[5];
rz(4.1604166) q[3];
rz(1.5570836) q[2];
rz(1.0214335) q[4];
rz(5.3601368) q[1];
cx q[4],q[0];
rz(2.5396021) q[5];
cx q[2],q[6];
rz(2.193251) q[3];
cx q[6],q[0];
cx q[2],q[5];
cx q[1],q[3];
rz(2.1924715) q[4];
rz(2.9431496) q[1];
cx q[0],q[4];
cx q[6],q[5];
cx q[3],q[2];
cx q[1],q[4];
cx q[0],q[5];
rz(5.4133053) q[6];
rz(4.406986) q[2];
rz(0.50993319) q[3];
rz(5.636134) q[4];
rz(5.2251168) q[6];
cx q[2],q[0];
cx q[1],q[5];
rz(5.9021776) q[3];
rz(2.8283003) q[4];
rz(0.29747458) q[0];
cx q[3],q[5];
rz(0.41037524) q[2];
cx q[6],q[1];
rz(4.0747232) q[5];
cx q[2],q[1];
rz(1.8160955) q[6];
cx q[4],q[0];
rz(2.3168431) q[3];
