m = csvread("output_17000.csv");
m = m(1:3,:);
d = zeros(1/0.001, 2);
d(:,1) = 0:0.001:1-0.001;
for i=1:size(d,1)-1
    indices = find((m(3,:) > d(i,1)) & (m(3,:) <= d(i+1,1)));
    d(i,2) = size(indices,2) / 17000;
end