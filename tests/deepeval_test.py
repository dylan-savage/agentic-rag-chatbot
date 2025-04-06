from dotenv import load_dotenv
load_dotenv() 

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric,ContextualRelevancyMetric,ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# 3 test cases corresponding to the 3 files uploaded to Qdrant
test_cases = [
    LLMTestCase(
        input="What is the Tush Push play in the NFl?",
        actual_output="The Tush Push play is a controversial tactic in the National Football League (NFL) where a player, typically a running back, is pushed or propelled forward by their teammates, often using their hands or bodies to generate additional momentum and gain extra yards. This play has been used by several teams over the years, but its legitimacy and fairness have been debated among coaches, players, and fans. The Tush Push play typically involves a running back who has been stopped or slowed down by a defender, and their teammates then push them forward to try and gain more yards or score a touchdown. While some argue that this play is a clever and creative way to gain an advantage, others see it as a form of cheating or exploiting a loophole in the rules. The controversy surrounding the Tush Push play has led to discussions about its impact on the game and whether it should be allowed or banned. Some argue that it is a safety concern, as it can lead to injuries or altercations between players, while others see it as a legitimate tactic that is within the rules of the game. The NFL has addressed the issue, and according to the league's rules, 'a player shall not push or block an opponent who is in the grasp of another player.' However, the interpretation and enforcement of this rule can be subjective, and it is often left to the discretion of the referees to decide whether a Tush Push play is legal or not. Despite the controversy, the Tush Push play remains a topic of interest and debate among football fans and analysts. Some teams have successfully used this play to gain an advantage, while others have been penalized for using it. Ultimately, the legitimacy and fairness of the Tush Push play will continue to be discussed and debated, and it will be up to the NFL to clarify and enforce its rules to ensure a safe and fair game for all players.",
        expected_output="The Tush Push is a quarterback sneak variation popularized by the Eagles, where players push the quarterback forward to gain short yardage. It has been successful but controversial due to safety and competitiveness concerns.",
        context=["The 'Tush Push' is a quarterback sneak tactic where teammates push the QB forward after the snap.",
                 "Popularized by the Eagles in 2022–2023 seasons.",
                 "Often used on 3rd/4th and 1-yard situations.",
                 "Has sparked NFL debate about player safety and potential rule changes."],
        retrieval_context=[
            "The 'Brotherly Shove': Why in vogue 'Tush Push' has become unstoppable play in the NFL",
            "In order to pass any motion, there needs to be a 75% majority in favor of the rule change among NFL owners, with 24 out of 32 votes needed.",
            "What is the tush push?",
            "The tush push rose to prominence a few years ago when the Eagles began deploying it in short-yardage situations.",
            "The combined efforts usually result in a short-yardage gain that is enough for either a first down or a touchdown and the Eagles' version of it is usually unstoppable.",
            "Eagles quarterback Jalen Hurts – who is the person with the ball in his hands and is being pushed from behind – has benefitted greatly from this play, with the majority of his 52 rushing touchdowns over the last four seasons coming from the tush push.",
            "It became a key driving factor in the team reaching the Super Bowl two years ago and in their title success last season.",
            "Like many other aspects across the NFL, other teams have tried to adopt the tush push with varying success, while the Eagles remain the masters of it.",
            "Why do teams want it outlawed?",
            "Despite the success of the tush push, it has become a controversial play, with some arguing it takes away competitiveness and makes football less exciting.",
            "The play, which bares similarities to the old-school quarterback sneak used in the early days of football, has also led to safety concerns, with players pushing against one another with all their force in such close proximity.",
            "Green Bay, which was beaten handily by the Eagles in the wild card round of the playoffs as Philadelphia went on to win Super Bowl LIX, was the team to table the motion to ban the play, with CEO and team president Mark Murphy saying the tush push was 'bad for the game.'",
            "'There is no skill involved and it is almost an automatic first down on plays of a yard or less,' Murphy added. 'We should go back to prohibiting the push of the runner.'",
            "Then you know the pushing of it adds the force piece, which again is, I think that exponentially raises my concern.",
            "What have people said about the play?",
            "As one of the more controversial plays in the NFL, public opinions appear divided on the future of the tush push.",
            "NFL announces changes to kickoffs, overtime, how the ball is spotted and instant replay but owners table 'tush push' ban",
            "By Ben Morse and Kyle Feldscher, CNN"
        ]
    ),
    LLMTestCase(
        input="Did Max verstappen win the 2025 Japan Grand Prix?",
        actual_output=" Yes, Max Verstappen won the 2025 Japan GP, beating McLaren's Lando Norris and Oscar Piastri at Suzuka. His victory was a result of a perfect weekend that included a scintillating qualifying lap and a composed race performance that kept the McLarens at arm's length. Verstappen's exceptional driving skills, strategic prowess, and the strong support of his team were key factors in his success. The win marked a significant milestone in the 2025 season, solidifying Verstappen's position as a top contender and setting the stage for an exciting conclusion to the championship.",
        expected_output="Max Verstappen won the 2025 Japan Grand Prix, dominating the race weekend and securing first place at Suzuka.",
        context=[
            "Max Verstappen won the 2025 Japanese Grand Prix at Suzuka.",
            "He secured pole position and led for most of the race.",
            "Red Bull's performance was praised despite recent car struggles.",
            "The win put Verstappen just one point behind Lando Norris in the drivers' standings.",
            "The race was seen as one of Verstappen's best performances.",
            "McLaren drivers Norris and Piastri finished behind him.",
            "Christian Horner and Alonso praised Verstappen's skill."
        ],
        retrieval_context=[
            "Verstappen's perfect Japan GP shows McLaren won't get easy ride",
            "Nate Saunders",
            "Apr 6, 2025, 09:00 AM ET",
            "Max Verstappen beat McLaren's Lando Norris and Oscar Piastri at Suzuka to go one point off the top of the drivers' championship.",
            "That rising reference point has ultimately been the crux of Red Bull's issues with the second car.",
            "Whatever word you used to frame Max Verstappen's Japanese Grand Prix victory -- and superlatives were in high supply throughout the weekend -- it was something special.",
            "Sunday's race was not a classic by any means, but Verstappen's weekend was a vintage of a different kind.",
            "It was a race win built on the bedrock of a scintillating qualifying lap that beat the benchmark McLarens to pole position on Saturday, and then the trademark pace and unshakeable composure which kept both orange cars at arm's length on Sunday, even when the consensus was they have the class-leading car in the F1 field.",
            "Norris on Verstappen incident: 'It was racing'",
            "SUZUKA, Japan -- Perfection. Magical. Inspirational.",
            "Speaking on the radio just after the checkered flag, long-time race engineer Gianpiero Lambiase succinctly summed up Verstappen's performance: 'That's perfection. Just perfection, Max. Fantastic.' It was hard to disagree."
        ]
    ),
    LLMTestCase(
        input="Did Duke win their 2025 March Madness matchup against Houston Men's basketball?",
        actual_output="Houston won the 2025 March Madness matchup against Duke Men's basketball after a stunning comeback. Duke had a 14-point lead but lost the game. The final score was not specified in the information provided, but it is clear that Houston won the game. For more information and the exact score, I recommend checking the official NCAA website or sports news websites for the latest updates on the 2025 March Madness tournament.",
        expected_output="No, Duke lost to Houston in the 2025 Final Four. Houston scored the last nine points to win 70-67.",
        context=[
            "Duke had a six-point lead in the final 35 seconds of the game.",
            "Houston scored the final nine points to win 70-67.",
            "Cooper Flagg missed a potential game-winning shot in the closing seconds.",
            "Flagg committed a foul that allowed Houston to take the lead.",
            "The game ended in a heartbreaking loss for Duke in the Final Four."
        ],
        retrieval_context=[
            "For Duke, stunned silence after epic collapse in Final Four",
            "Pete Thamel",
            "Apr 6, 2025, 03:27 AM ET",
            "Houston's defenders were their marauding selves all night, with the most jarring statistic in the box score being that of Duke center Khaman Maluach when he failed to grab a rebound in more than 21 minutes of play and ending the night with a plus-minus of -20.",
            "Roberts' final salvo was getting a tough contest on Flagg's potential game winner.",
            "'I thought he did an awesome job of getting his hands up high enough that it wasn't an easy look,' Sampson said of Roberts. 'Some tough shots all night.'",
            "Flagg finished the contest with 27 points, shooting 8-for-19 from the field. He got little help, as Duke had only one field goal over the game's last 10:30.",
            "He rode back to the Duke locker room in a golf cart at 11:54 p.m., staring into space with a towel wrapped around his neck. Flagg entered the cone of silence suddenly facing the end of a season and likely a college career.",
            "Three minutes later, Duke coach Jon Scheyer rode past with his wife next to him and athletic director Nina King sitting in the back. After leading by as much as 14, Duke had just coughed up the fifth-biggest lead in Final Four history. The loss will echo, just like that slamming door, long into the offseason.",
            "'I keep going back, we're up six with under a minute to go,' Scheyer said.",
            "'We just have to finish the deal.'",
            "Tougher to explain was Flagg's over-the-back foul on Roberts when Duke's Tyrese Proctor missed the front end of a one-and-one with 20 seconds remaining. Duke led 67-66 at the time, and Flagg got whistled for a foul on Roberts, who clearly had Flagg boxed out.",
            "Cooper Flagg, who finished with 27 points, said his missed 12-foot jumper with Duke trailing by one in the closing seconds Saturday night was a shot he is 'willing to live with in the scenario.'",
            "The validity of the call will long be debated on barstools at the Final Four, but Flagg put himself and Duke in a vulnerable position by appearing to hold down Roberts' left arm and getting whistled for it.",
            "Roberts, a 63% free throw shooter, changed the game by making both ends of the one-and-one, pushing Houston to a 68-67 lead and setting the stage for Flagg's final foray.",
            "For a program that holds a defiant image of grit and toughness, it's fitting that Houston's trip to the national title game featured a game-changing boxout. Kellen Sampson, the Houston assistant and son of Cougars coach Kelvin Sampson, broke out one of his father's folksy basketball sayings to sum up the moment.",
            "'Discipline gets you beat more than great helps you win,' Kellen Sampson said. 'I've probably heard it a hundred million times growing up. Look, the more disciplined you are, the more that you can find yourself doing little tiny things that's going to win.'",
            "'A big-time free throw blockout was exactly what was needed,' he added.",
            "Regardless of any debate over the call, Flagg's foul put Duke in a suddenly unthinkable position. The Blue Devils went from a six-point lead with 35 seconds left to trailing by one at the 19-second mark. The foul was the final swing: up one to down one.",
            "The key for Houston came from leaving Roberts alone on Flagg, something it didn't do early in the game. Flagg picked the Cougars apart with his passing, and they made an adjustment to let Roberts handle the matchup by himself.",
            "'We said here at halftime we're going to trust J'Wan,' Sampson said.",
            "Flagg's missed 12-foot jumper, with Duke trailing by one point, will be the play that will live forever in replays. Duke had a chance to take control of the game and stop the hemorrhaging; a timeout was called with 17 seconds left. The Blue Devils cleared out for Flagg, who got an isolation matchup with Houston sixth-year senior J'Wan Roberts. Flagg pulled up from inside the lane and faded away from the outstretched arms of the 6-foot-8 Roberts. The shot caromed off the front rim.",
            "'It's the play Coach drew up,' Flagg said. 'Took it into the paint. Thought I got my feet set, rose up. Left it short, obviously. A shot I'm willing to live with in the scenario.'",
            "There was no second-guessing the play or the look. It simply didn't go in.",
            "But even after a spree of inbounds failures, misses and mental gaffes, two key moments in the final 20 seconds from star freshman Cooper Flagg -- a foul and a miss -- capped the stunning meltdown.",
            "Comeback Cougars: Houston rallies, shocks Duke"
        ]
    )
]

# Define evaluation metrics
metrics = [
    AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
    FaithfulnessMetric(model="gpt-4o-mini", threshold=0.7),
    ContextualPrecisionMetric(model="gpt-4o-mini", threshold=0.7),
    ContextualRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
    ContextualRecallMetric(model="gpt-4o-mini", threshold=0.7)
]

# Run the tests
for test_case in test_cases:
    # Manually evaluate
    for metric in metrics:
        metric.measure(test_case)
        print(f"\n {metric.__class__.__name__}")
        print(f"Score: {metric.score}")
        if hasattr(metric, "reason"):
            print(f"Reason: {metric.reason}")

