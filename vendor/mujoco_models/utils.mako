<%def name="make_maze(structure, height, size_scaling)">
    % for i in xrange(len(structure)):
        % for j in xrange(len(structure[0])):
            % if str(structure[i][j]) == '1':
                <geom
                  name="block_${i}_${j}"
                  pos='${j*size_scaling} ${i*size_scaling} ${height/2*size_scaling}'
                  size='${0.5*size_scaling} ${0.5*size_scaling} ${height/2*size_scaling}'
                  type='box'
                  material=""
                  contype="1"
                  conaffinity="1"
                  rgba='0.4 0.4 0.4 1'
                  />
            % endif
        % endfor
    % endfor
</%def>

<%def name="make_contacts(geom_name, structure)">
    % for i in xrange(len(structure)):
        % for j in xrange(len(structure[0])):
            % if str(structure[i][j]) == '1':
                <pair
                  geom1="${geom_name}"
                  geom2="block_${i}_${j}"
                  />
            % endif
        % endfor
    % endfor
</%def>

<%def name="find_goal_range(structure, size_scaling)">
    <%
        found = False
        goal_range = []
        for i in xrange(len(structure)):
            for j in xrange(len(structure[0])):
                if structure[i][j] == 'g':
                    goal_range.append(j*size_scaling-size_scaling*0.5),
                    goal_range.append(j*size_scaling+size_scaling*0.5),
                    goal_range.append(i*size_scaling-size_scaling*0.5),
                    goal_range.append(i*size_scaling+size_scaling*0.5),
                    found = True
                    break
            if found:
                break
    %>
    <numeric name="goal_range" data="${" ".join(map(str, goal_range))}" />
</%def>

<%def name="find_robot(structure, size_scaling, z_offset=0)">
    <%
        robot_pos = [0, 0, z_offset]
        found = False
        for i in xrange(len(structure)):
            for j in xrange(len(structure[0])):
                if structure[i][j] == 'r':
                    robot_pos[0] = j*size_scaling
                    robot_pos[1] = i*size_scaling
                    found = True
                    break
            if found:
                break
    %>
    ${' '.join(map(str, robot_pos))}
</%def>

<%def name="encode_map(structure, size_scaling)">
    <%
        data = []
        data.append(len(structure))
        data.append(len(structure[0]))
        data.append(size_scaling)
        for i in xrange(len(structure)):
            for j in xrange(len(structure[0])):
                if structure[i][j] == 1:
                    data.append(1)
                elif structure[i][j] == 'g':
                    data.append(2)
                else:
                    data.append(0)
    %>
    ${' '.join(map(str, data))}
</%def>
