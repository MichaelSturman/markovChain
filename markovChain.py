import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# https://www.mysense.ai/2021/07/14/markov-chain-applications-in-data-science/
# Attempt to replicate the epidemioloy example

# Initial simple example
# Three states: susceptible (0), infected (1) and recovered (2).
# An individual can move from susceptible to infected, but not to recovered
# An individual can move from infected to recovered, but not to susceptible
# An individual can move from recovered to infected but not to susceptible

def simpleMarkov(pInf, pRec, pReInf, numPeople, numDays):
    P = [[1 - pInf, pInf, 0], [0, 1 - pRec, pRec], [0, 0, 1 - pReInf]]

    # Each susceptible individual has a 100*pInf% chance of catching the disease on a
    # given time step.
    # An infected individual has a 100*pRec% daily chance of recovery
    # A recovered individual has a 100*pReInf% of reinfection

    cols = [0, 1, 2]
    people = pd.DataFrame(columns=cols)

    sus = []
    inf = []
    rec = []

    for day in range(0, numDays):
        # On day 0 set up starting conditions - all healthy
        if day == 0:
            for i in range(0, numPeople):
                people.loc[len(people.index)] = [1, 0, 0]

        # On subsequent days update the conditions
        else:
            for p in range(0, numPeople):
                # Generate random number between 0 and 1
                prob = np.random.rand(1)
                # Adding updated marker. Code was previously looking to skip infected
                # status and move into recovery. Updated shows that in this timestep
                # we have already changed the state of the person.
                updated = 0
                if people.loc[p][0] == 1:
                    # Person remains uninfected
                    if prob < (1 - pInf):
                        pass
                    # Person is infected
                    else:
                        people.loc[p] = [0, 1, 0]
                        updated = 1

                if people.loc[p][1] == 1 and updated == 0:
                    # Person reamins infected
                    if prob < (1 - pRec):
                        pass
                    # person recovers
                    else:
                        people.loc[p] = [0, 0, 1]
                        updated = 1

                if people.loc[p][2] == 1 and updated == 0:
                    # Person remains uninfected
                    if prob < (1 - pReInf):
                        pass
                    # person is reinfected
                    else:
                        people.loc[p] = [0, 1, 0]
                        updated = 1

        sus.append(sum(people[0]))
        inf.append(sum(people[1]))
        rec.append(sum(people[2]))

    fig, ax = plt.subplots()

    ax.plot(sus, color='b', linestyle='dashed', linewidth=1, label='Susceptible')
    ax.plot(inf, color='r', linestyle='dashed', linewidth=1, label='Infected')
    ax.plot(rec, color='g', linestyle='dashed', linewidth=1, label='Recovered')
    legend = ax.legend(loc='lower right')
    plt.ylabel('Number of People')
    plt.xlabel('Number of Days')
    plt.show()

    return 0

def markovWithMortality(pInf, pRec, pReInf, pDeath, numPeople, numDays):
    # Adding in a fourth state - deceased. Absorbing state (for obvious reasons)
    # which can only be reached from infected.

    # State 0: susceptible, State 1: infected, State 2: recovered, State 3: deceased
    P = [[1 - pInf, pInf, 0, 0], [0, 1 - (pRec + pDeath), pRec, pDeath], [0, 0, 1 - pReInf, 0], [0, 0, 0, 1]]

    cols = [0, 1, 2, 3]
    people = pd.DataFrame(columns=cols)

    sus = []
    inf = []
    rec = []
    dea = []

    for day in range(0, numDays):
        if day == 0:
            for i in range(0, numPeople):
                people.loc[len(people.index)] = [1, 0, 0, 0]
        else:
            for p in range(0, len(people)):

                prob = np.random.rand(1)
                updated = 0

                if people.loc[p][0] == 1:
                    if prob < 1 - pInf:
                        pass
                    else:
                        people.loc[p] = [0, 1, 0, 0]
                        updated = 1

                if people.loc[p][1] == 1 and updated == 0:
                    if prob < (1 - (pRec + pDeath)):
                        pass
                    if prob >= 1 - (pRec + pDeath) and prob < 1 - pDeath:
                        people.loc[p] = [0, 0, 1, 0]
                        updated = 1
                    if prob >= 1 - pDeath:
                        people.loc[p] = [0, 0, 0, 1]
                        updated = 1


                if people.loc[p][2] == 1 and updated == 0:
                    if prob < 1 - pReInf:
                        pass
                    else:
                        people.loc[p] = [0, 1, 0, 0]
                        updated = 1

                # No need to do anything for state 3 (deceased)

        sus.append(sum(people[0]))
        inf.append(sum(people[1]))
        rec.append(sum(people[2]))
        dea.append(sum(people[3]))


    fig, ax = plt.subplots()

    ax.plot(sus, color='b', linestyle='dashed', linewidth=1, label='Susceptible')
    ax.plot(inf, color='r', linestyle='dashed', linewidth=1, label='Infected')
    ax.plot(rec, color='g', linestyle='dashed', linewidth=1, label='Recovered')
    ax.plot(dea, color='grey', linestyle='dashed', linewidth=1, label='Deceased')
    legend = ax.legend(loc='lower right')
    plt.ylabel('Number of People')
    plt.xlabel('Number of Days')
    plt.show()


    return 0

# Issue with the above model as that if we run it to infinity, eventually everyone
# will die. This is because the probability of infection is constant. Would make
# more sense if it was proportional to the number of infected people.Attempted below

def markovWithMortalityPInfProbNumInf(pInfMax, pRec, pReInfMax, pDeath, numPeople, numInfected, numDays):
    # Adding in a fourth state - deceased. Absorbing state (for obvious reasons)
    # which can only be reached from infected.

    # State 0: susceptible, State 1: infected, State 2: recovered, State 3: deceased

#    P = [[1 - (pInfMax*numInfected/(numPeople - numDeceased)), pInfMax*numInfected/(numPeople - numDeceased), 0, 0],
#        [0, 1 - (pRec + pDeath), pRec, pDeath],
#        [0, 0, 1 - pReInfMax*numInfected/(numPeople - numDeceased), 0], [0, 0, 0, 1]]

    cols = [0, 1, 2, 3]
    people = pd.DataFrame(columns=cols)

    sus = []
    inf = []
    rec = []
    dea = []

    for day in range(0, numDays):
        if day == 0:
            for i in range(0, numPeople - numInfected):
                people.loc[len(people.index)] = [1, 0, 0, 0]
            for i in range(numPeople - numInfected, numPeople):
                people.loc[len(people.index)] = [0, 1, 0, 0]
        else:
            for p in range(0, len(people)):

                prob = np.random.rand(1)
                updated = 0

                if people.loc[p][0] == 1:
                    if prob < 1 - pInfMax*numInfected/(numPeople - sum(people[3])):
                        pass
                    else:
                        people.loc[p] = [0, 1, 0, 0]
                        updated = 1

                if people.loc[p][1] == 1 and updated == 0:
                    if prob < (1 - (pRec + pDeath)):
                        pass
                    if prob >= 1 - (pRec + pDeath) and prob < 1 - pDeath:
                        people.loc[p] = [0, 0, 1, 0]
                        updated = 1
                    if prob >= 1 - pDeath:
                        people.loc[p] = [0, 0, 0, 1]
                        updated = 1


                if people.loc[p][2] == 1 and updated == 0:
                    if prob < 1 - pReInfMax*numInfected/(numPeople - sum(people[3])):
                        pass
                    else:
                        people.loc[p] = [0, 1, 0, 0]
                        updated = 1


                # No need to do anything for state 3 (deceased)
        numInfected = sum(people[3])

        sus.append(sum(people[0]))
        inf.append(sum(people[1]))
        rec.append(sum(people[2]))
        dea.append(sum(people[3]))


    fig, ax = plt.subplots()

    ax.plot(sus, color='b', linestyle='dashed', linewidth=1, label='Susceptible')
    ax.plot(inf, color='r', linestyle='dashed', linewidth=1, label='Infected')
    ax.plot(rec, color='g', linestyle='dashed', linewidth=1, label='Recovered')
    ax.plot(dea, color='grey', linestyle='dashed', linewidth=1, label='Deceased')
    legend = ax.legend(loc='lower right')
    plt.ylabel('Number of People')
    plt.xlabel('Number of Days')
    plt.show()

    return 0

# Would also make sense if recovered people have a lower chance of death.
# Could include in this effort a chance of moving from susceptible to Immune (vaccination)
# Incorporate a second group - vaccinated/recovered. This group will have a
# lower chance of contracting the disease and a lower chance of dying from it.
# It doesn't make sense to define a "vaccine probability". Makes more sense to
# have a rate/day of vaccines that can be given.

def markovVaccine(pInfMax, pRec, pDeath, vacRate, numPeople, numInfected, numDays):

    # Define two groups - people with no immunity and people with immunity.
    # People with immunity include vaccinated and recovered individuals and have
    # infection and death probabilities .05* that of those with no immunity. We can do
    # away with the "recovered" category and only have un-infected, infected and
    # dead.

    # vacRate defined as the number of vaccines given per day. To be eligible for
    # vaccination an individual must not be infected.

    # State 0: healthy
    # State 1: ill
    # State 2: dead

    cols = [0, 1, 2]
    peopleNoImmunity = pd.DataFrame(columns=cols)
    peopleImmunity = pd.DataFrame(columns=cols)

    sus = []
    inf = []
    imm = []
    dea = []

    for day in range(0, numDays):
        if day == 0:
            for i in range(0, numPeople - numInfected):
                peopleNoImmunity.loc[len(peopleNoImmunity.index)] = [1, 0, 0]
            for i in range(numPeople - numInfected, numPeople):
                peopleNoImmunity.loc[len(peopleNoImmunity.index)] = [0, 1, 0]

        else:
            for q in range(0, len(peopleImmunity)):

                probSCa = np.random.rand(1)
                updated = 0

                if peopleImmunity.loc[q][0] == 1:
                    if probSCa < 1 - 0.05*pInfMax*numInfected/(numPeople - (sum(peopleImmunity[2]) + sum(peopleNoImmunity[2]))):
                        pass
                    else:
                        peopleImmunity.loc[q] = [0, 1, 0]
                        updated = 1

                if peopleImmunity.loc[q][1] == 1 and updated == 0:
                    if probSCa < 1 - (pRec + 0.05*pDeath):
                        pass
                    if probSCa >= 1 - (pRec + 0.05*pDeath) and probSCa < 1 - 0.05*pDeath:
                        peopleImmunity.loc[q] = [1, 0, 0]
                        updated = 1
                    if probSCa >= 1 - 0.05*pDeath:
                        peopleImmunity.loc[q] = [0, 0, 0]
                        updated = 1

            # This second loop was a bit harded - we have to drop out the occasional
            # point and add it into the immune group. Resolved by using a while
            # loop with p, which we add one to every time we pass through the loop.
            # In situations where we drop a point and reindex the array, point 6
            # moves to point 5 but without modification, p would pass to 6 without
            # consideration. This would mean we would not update point 6 (now point
            # 5), and also that we overshoot our index. For this reason, we subtract
            # and add 1 to p for drop cases, leaving it the same and allowing us
            # to update the new occupant of that index

            p = 0

            while p < len(peopleNoImmunity):

                probSCb = np.random.rand(1)
                probVac = np.random.rand(1)
                updated = 0

                if peopleNoImmunity.loc[p][0] == 1:
                    if probSCb < 1 - pInfMax*numInfected/(numPeople - (sum(peopleImmunity[2]) + sum(peopleNoImmunity[2]))):
                        if probVac >= 1 - vacRate/sum(peopleNoImmunity[0]):
                            peopleImmunity.loc[len(peopleImmunity.index)] = [1, 0, 0]
                            peopleNoImmunity = peopleNoImmunity.drop(p).reset_index(drop=True)
                            updated = 1
                            if p!=0:
                                p -= 1
                        else:
                            pass
                    else:
                        peopleNoImmunity.loc[p] = [0, 1, 0]
                        updated = 1

                if peopleNoImmunity.loc[p][1] == 1 and updated == 0:
                    if probSCb < (1 - (pRec + pDeath)):
                        pass
                    if probSCb >= 1 - (pRec + pDeath) and probSCb < 1 - pDeath:
                        peopleImmunity.loc[len(peopleImmunity.index)] = [1, 0, 0]
                        peopleNoImmunity = peopleNoImmunity.drop(p).reset_index(drop=True)
                        updated = 1
                        if p!=0:
                            p -= 1
                    if probSCb >= 1 - pDeath:
                        peopleNoImmunity.loc[p] = [0, 0, 1]
                        updated = 1

                p+=1

            numSusceptible = sum(peopleNoImmunity[0])
            numInfected = sum(peopleNoImmunity[1]) + sum(peopleImmunity[1])
            numImmune = sum(peopleImmunity[0])
            numDeceased = sum(peopleNoImmunity[2]) + sum(peopleImmunity[2])

            sus.append(numSusceptible)
            inf.append(numInfected)
            imm.append(numImmune)
            dea.append(numDeceased)

    fig, ax = plt.subplots()

    ax.plot(sus, color='b', linestyle='dashed', linewidth=1, label='Susceptible')
    ax.plot(inf, color='r', linestyle='dashed', linewidth=1, label='Infected')
    ax.plot(imm, color='g', linestyle='dashed', linewidth=1, label='Immune')
    ax.plot(dea, color='grey', linestyle='dashed', linewidth=1, label='Deceased')
    legend = ax.legend(loc='lower right')
    plt.ylabel('Number of People')
    plt.xlabel('Number of Days')
    plt.show()

    return 0

# Final addition to this model - diseases are not static, they mutate which can
# effect the transmissability/mortality rate as well as the resistance offered
# by existing immunity. We will ignore any changes in overall infection/mortality
# probabilities, but a mutation will mean that chance of "immune" people catching
# or dying from the disease will go up to 90% (make it dynamic) of the susceptible.

# New variables below: mutMod - protection from immunity. Start at .05, markovMutation
# will change this. pMut - chance of a mutation taking place. WIll start with it
# as static, but makes more sense to have it as proportional to the number of
# infected individuals

def markovMutation(pInfMax, pRec, pDeath, pMut, imPro, vacRate, numPeople, numInfected, numDays):

    # State 0: healthy
    # State 1: ill
    # State 2: dead

    cols = [0, 1, 2]
    peopleNoImmunity = pd.DataFrame(columns=cols)
    peopleImmunity = pd.DataFrame(columns=cols)

    inf = []
    uninf = []
    dea = []

    for day in range(0, numDays):

        mutCheck = np.random.rand(1)
        if mutCheck >= 1 - pMut:
            # A mutation resets susceptibility/immunity to the start point. 
            print("mutation!", day)
            peopleNoImmunity = peopleNoImmunity.append(peopleImmunity, ignore_index=True)
            n = len(peopleImmunity)
            peopleImmunity.drop(index=peopleImmunity.index[:n], inplace=True)


        if day == 0:
            for i in range(0, numPeople - numInfected):
                peopleNoImmunity.loc[len(peopleNoImmunity.index)] = [1, 0, 0]
            for i in range(numPeople - numInfected, numPeople):
                peopleNoImmunity.loc[len(peopleNoImmunity.index)] = [0, 1, 0]

        else:
            for q in range(0, len(peopleImmunity)):

                probSCa = np.random.rand(1)
                updated = 0

                if peopleImmunity.loc[q][0] == 1:
                    if probSCa < 1 - imPro*pInfMax*numInfected/(numPeople - (sum(peopleImmunity[2]) + sum(peopleNoImmunity[2]))):
                        pass
                    else:
                        peopleImmunity.loc[q] = [0, 1, 0]
                        updated = 1

                if peopleImmunity.loc[q][1] == 1 and updated == 0:
                    if probSCa < 1 - (pRec + imPro*pDeath):
                        pass
                    if probSCa >= 1 - (pRec + imPro*pDeath) and probSCa < 1 - imPro*pDeath:
                        peopleImmunity.loc[q] = [1, 0, 0]
                        updated = 1
                    if probSCa >= 1 - 0.05*pDeath:
                        peopleImmunity.loc[q] = [0, 0, 0]
                        updated = 1

            # This second loop was a bit harded - we have to drop out the occasional
            # point and add it into the immune group. Resolved by using a while
            # loop with p, which we add one to every time we pass through the loop.
            # In situations where we drop a point and reindex the array, point 6
            # moves to point 5 but without modification, p would pass to 6 without
            # consideration. This would mean we would not update point 6 (now point
            # 5), and also that we overshoot our index. For this reason, we subtract
            # and add 1 to p for drop cases, leaving it the same and allowing us
            # to update the new occupant of that index

            p = 0

            while p < len(peopleNoImmunity):

                probSCb = np.random.rand(1)
                probVac = np.random.rand(1)
                updated = 0

                if peopleNoImmunity.loc[p][0] == 1:
                    if probSCb < 1 - pInfMax*numInfected/(numPeople - (sum(peopleImmunity[2]) + sum(peopleNoImmunity[2]))):
                        if probVac >= 1 - vacRate/sum(peopleNoImmunity[0]):
                            peopleImmunity.loc[len(peopleImmunity.index)] = [1, 0, 0]
                            peopleNoImmunity = peopleNoImmunity.drop(p).reset_index(drop=True)
                            updated = 1
                            if p!=0:
                                p -= 1
                        else:
                            pass
                    else:
                        peopleNoImmunity.loc[p] = [0, 1, 0]
                        updated = 1

                if peopleNoImmunity.loc[p][1] == 1 and updated == 0:
                    if probSCb < (1 - (pRec + pDeath)):
                        pass
                    if probSCb >= 1 - (pRec + pDeath) and probSCb < 1 - pDeath:
                        peopleImmunity.loc[len(peopleImmunity.index)] = [1, 0, 0]
                        peopleNoImmunity = peopleNoImmunity.drop(p).reset_index(drop=True)
                        updated = 1
                        if p!=0:
                            p -= 1
                    if probSCb >= 1 - pDeath:
                        peopleNoImmunity.loc[p] = [0, 0, 1]
                        updated = 1

                p+=1

            numSusceptible = sum(peopleNoImmunity[0])
            numInfected = sum(peopleNoImmunity[1]) + sum(peopleImmunity[1])
            numImmune = sum(peopleImmunity[0])
            numDeceased = sum(peopleNoImmunity[2]) + sum(peopleImmunity[2])


            uninf.append(numSusceptible + numImmune)
            inf.append(numInfected)
            dea.append(numDeceased)

    fig, ax = plt.subplots()

    ax.plot(uninf, color='g', linestyle='dashed', linewidth=1, label='Uninfected')
    ax.plot(inf, color='r', linestyle='dashed', linewidth=1, label='Infected')
    ax.plot(dea, color='grey', linestyle='dashed', linewidth=1, label='Deceased')
    legend = ax.legend(loc='lower right')
    plt.ylabel('Number of People')
    plt.xlabel('Number of Days')
    plt.show()

    return 0


def main():
    #simpleMarkov(pInf, pRec, pReInf, numPeople, numDays)
    #simpleMarkov(0.2, 0.1, 0.05, 1000, 100)

    #markovWithMortality(pInf, pRec, pReInf, pDeath, numPeople, numDays):
    #markovWithMortality(0.2, 0.1, 0.05, 0.01, 1000, 50)

    #markovWithMortalityPInfProbNumInf(pInfMax, pRec, pReInfMax, pDeath, numPeople, numInfected, numDays)
    #markovWithMortalityPInfProbNumInf(0.4, 0.1, 0.01, 0.01, 1000, 350, 200)

    #markovVaccine(pInfMax, pRec, pDeath, vacRate, numPeople, numInfected, numDays):
    #markovVaccine(0.4, 0.1, 0.01, 15, 4000, 700, 50)

    #markovMutation(pInfMax, pRec, pDeath, pMut, imPro, vacRate, numPeople, numInfected, numDays):
    markovMutation(0.4, 0.1, 0.01, 0.02, 0.05, 15, 1000, 350, 100)



if __name__ == '__main__':
    main()
