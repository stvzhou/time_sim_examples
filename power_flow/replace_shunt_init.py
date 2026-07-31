with open("/usr/local/google/home/sxzhou/Downloads/ACTIVSg2000.RAW", "r") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    new_lines.append(lines[i])
    if "BEGIN SWITCHED SHUNT DATA" in lines[i]:
        i += 1
        break
    i += 1

while i < len(lines):
    if "END OF SWITCHED SHUNT DATA" in lines[i]:
        new_lines.append(lines[i])
        break
    fields = lines[i].split(",")
    fields[-1] = fields[-3]
    new_lines.append(",".join(fields) + "\n")
    i += 1


with open("/usr/local/google/home/sxzhou/Downloads/temp.RAW", "w") as f:
    f.writelines(new_lines)
