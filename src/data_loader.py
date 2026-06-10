import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'UpdatedResumeDataSet.csv')

SAMPLE_RESUMES = {
    'Data Science': [
        "Experienced data scientist with expertise in Python, machine learning, TensorFlow, Keras, pandas, numpy, scikit-learn. Built predictive models for classification and regression. Proficient in SQL, data visualization with matplotlib and seaborn. Published research on deep learning and NLP.",
        "Data scientist with 4 years experience in statistical modeling, A/B testing, feature engineering, and deploying ML pipelines. Used XGBoost, LightGBM, and neural networks. Strong in Python, R, SQL. Experience with AWS SageMaker and MLflow.",
        "Machine learning engineer skilled in computer vision, NLP, time series forecasting. Developed production ML systems at scale using PyTorch and TensorFlow. Knowledge of MLOps, Docker, Kubernetes. PhD in computer science.",
        "Junior data scientist with strong Python skills, built regression and classification models using scikit-learn. Familiar with data wrangling using pandas, visualization using matplotlib. Completed multiple Kaggle competitions. BSc Statistics.",
        "Senior data scientist leading a team of 5, developed recommendation systems, churn prediction, and fraud detection models. Expertise in PySpark, Hadoop, and big data technologies alongside deep learning frameworks.",
        "Data analyst transitioning to data science, strong in SQL, Python, and Tableau. Built dashboards and predictive models for retail analytics. Familiar with sklearn, linear regression, and logistic regression. MBA with analytics focus.",
        "Research scientist with publications in machine learning and NLP. Expert in transformers, BERT, GPT-style models, and fine-tuning LLMs. Skilled in HuggingFace, PyTorch, and experiment tracking with WandB. 3 patents in AI.",
        "Data scientist at fintech startup, built fraud detection and credit scoring models. Proficient in feature engineering, SHAP values, model interpretability, and deploying models via FastAPI. Experience with AWS Lambda and S3.",
        "Applied ML engineer focusing on computer vision projects: object detection with YOLO, image segmentation with U-Net, and GANs for data augmentation. Proficient in OpenCV, PyTorch, and CUDA optimization.",
        "Quantitative analyst with strong background in statistics, probability, and machine learning. Built algorithmic trading strategies using ML models. Expert in Python, R, pandas, and financial time series analysis.",
        "Data science manager overseeing a team building customer lifetime value models and personalization engines. Expert in A/B testing frameworks, causal inference, and translating business problems into ML solutions.",
        "NLP engineer specializing in text classification, named entity recognition, and sentiment analysis. Built chatbots and information extraction pipelines. Proficient in spaCy, NLTK, transformers, and Elasticsearch.",
        "ML ops engineer setting up end-to-end ML pipelines with Airflow, Kubeflow, and MLflow. Automated model training, evaluation, and deployment on Kubernetes. Strong in Python, Docker, and CI/CD for ML.",
        "Data scientist in healthcare domain building clinical prediction models, survival analysis, and patient risk stratification. Experience with EHR data, HIPAA compliance, and medical image analysis using deep learning.",
        "Kaggle grandmaster with top 1% rankings in tabular, NLP, and vision competitions. Expert in ensemble methods, stacking, gradient boosting (XGBoost, LightGBM, CatBoost), and cross-validation strategies.",
    ],
    'Java Developer': [
        "Java developer with 5 years experience building enterprise applications using Spring Boot, Hibernate, and Microservices. Proficient in REST API design, Maven, JUnit testing, and CI/CD pipelines. Experience with Oracle and PostgreSQL databases.",
        "Full-stack Java developer skilled in Spring MVC, Angular, HTML/CSS/JavaScript. Built scalable backend systems. Used RabbitMQ and Kafka for messaging. Familiar with Docker, AWS, and Git. Agile/Scrum practitioner.",
        "Java backend developer experienced in building RESTful web services, working with SQL/NoSQL databases (MySQL, MongoDB). Proficient in JPA, Hibernate, Spring Security, and JWT authentication. Contributed to open source projects.",
        "Senior Java architect with expertise in distributed systems, design patterns, and high-performance applications. Deep knowledge of JVM internals, multithreading, and concurrency. Mentored junior developers and conducted code reviews.",
        "Java developer with expertise in Android mobile application development using Java and Kotlin. Published 5 apps on Google Play. Familiar with Firebase, Retrofit, Room database, and MVVM architecture.",
        "Java EE developer with expertise in building banking applications. Proficient in EJB, JMS, JNDI, and Oracle WebLogic. Strong in transactional systems, ACID properties, and financial data processing. OCPJP certified.",
        "Microservices architect designing Java-based distributed systems using Spring Cloud, Eureka, Zuul, and Feign. Expertise in fault tolerance with Hystrix, distributed tracing with Zipkin, and API gateway patterns.",
        "Java developer specializing in performance optimization, JVM tuning, GC analysis, and profiling with JProfiler and VisualVM. Reduced application latency by 40%. Expert in Java concurrency, thread pools, and async processing.",
        "Java automation test engineer using Selenium, TestNG, and Cucumber for BDD. Built robust test frameworks for web applications. Proficient in Maven, Jenkins CI, and Allure reporting. ISTQB certified.",
        "Spring Boot developer building cloud-native microservices deployed on Kubernetes. Experience with service mesh (Istio), config management, and 12-factor app principles. Strong DevOps integration skills.",
        "Java developer in telecom domain building billing systems and OSS/BSS applications. Experience with XML processing, web services (SOAP/REST), and large-scale data migration projects. 8 years of industry experience.",
        "Full-stack developer using Java backend with React frontend. Built e-commerce platforms handling 1M+ transactions. Proficient in Spring Boot, JPA, Redis caching, Elasticsearch, and AWS deployment.",
        "Java developer with big data experience using Apache Spark Java API, Hadoop MapReduce, and Hive. Built ETL pipelines processing terabytes of data daily. Expert in HDFS, YARN, and cluster administration.",
        "Senior Java developer leading API development for a SaaS platform. Expert in OpenAPI/Swagger documentation, versioning strategies, rate limiting, and OAuth2 integration. 7 years experience in product companies.",
        "Java developer with deep knowledge of design patterns: Singleton, Factory, Observer, Strategy, and Builder. Wrote clean, SOLID code with 90%+ unit test coverage. Experienced in code review and technical documentation.",
    ],
    'Python Developer': [
        "Python developer with 3 years building web applications using Django and Flask. Experienced in REST APIs, PostgreSQL, Redis caching, Celery task queues. Proficient in Docker, Git, and Linux. Wrote comprehensive unit tests.",
        "Python backend engineer with expertise in FastAPI, SQLAlchemy, and async programming. Built high-throughput data pipelines using Apache Kafka and Airflow. Strong in algorithms and data structures. Contributed to open-source.",
        "Python developer specializing in automation scripting, web scraping with BeautifulSoup and Scrapy, and data processing. Built ETL pipelines. Familiar with AWS Lambda and serverless architectures. Solid Git workflow.",
        "Full-stack Python developer skilled in Django REST Framework, React frontend, and DevOps practices. Deployed applications on AWS and GCP. Set up CI/CD with GitHub Actions. Experienced in microservices architecture.",
        "Python developer focused on scientific computing, simulation, and numerical methods using NumPy, SciPy, and Matplotlib. Experience in research environments, data analysis, and visualization for academic papers.",
        "Python automation engineer building test frameworks with pytest, Robot Framework, and Selenium. Created CI/CD pipelines with Jenkins and GitLab. Expert in API testing with requests library and mock testing.",
        "Senior Python developer building fintech APIs handling millions of transactions. Expert in async Python with asyncio and aiohttp, database query optimization, and Redis for session management and rate limiting.",
        "Python data engineer designing and maintaining data warehouses. Proficient in Apache Spark PySpark, dbt, Airflow DAGs, and Snowflake. Built real-time streaming pipelines with Kafka and Flink. AWS certified.",
        "Python developer with expertise in building CLI tools and developer utilities. Published 10+ packages on PyPI. Strong in packaging, virtual environments, type hints, mypy static analysis, and documentation with Sphinx.",
        "Backend Python developer for an e-learning platform. Built video streaming APIs, payment integrations (Stripe, PayPal), and notification systems. Expert in Django ORM optimization, caching strategies, and horizontal scaling.",
        "Python developer specializing in cybersecurity tools: vulnerability scanners, log analyzers, and network monitoring scripts. Familiar with Scapy, paramiko, and cryptography library. Contributed to bug bounty programs.",
        "Python web developer with Django experience in multi-tenant SaaS applications. Built custom authentication backends, permission systems, and admin customizations. Expert in Django signals, middleware, and custom management commands.",
        "Python ML developer bridging data science and production. Converted Jupyter notebooks into deployable FastAPI services. Expert in model serialization, A/B testing infrastructure, and feature stores.",
        "Junior Python developer passionate about clean code and testing. Built REST APIs with Flask, practiced TDD with pytest, and learned Docker containerization. BSc Computer Science. Active open-source contributor.",
        "Python developer building IoT data collection systems using MQTT, InfluxDB, and Grafana. Automated data ingestion from 500+ sensors. Experience with Raspberry Pi, edge computing, and time-series databases.",
    ],
    'HR': [
        "HR Manager with 7 years experience in talent acquisition, employee relations, performance management, and HR policy development. Proficient in HRIS systems (SAP HR, Workday). Conducted training programs and led teams of 10.",
        "Recruitment specialist with expertise in sourcing candidates through LinkedIn, job portals, and campus hiring. Managed end-to-end hiring for technical and non-technical roles. Skilled in ATS tools and behavioral interviews.",
        "HR Business Partner experienced in organizational development, change management, compensation benchmarking, and workforce planning. Aligned HR strategy with business objectives. MBA in Human Resources.",
        "HR Generalist managing payroll processing, benefits administration, onboarding/offboarding, and compliance with labor laws. Handled grievance procedures and disciplinary actions. Maintained accurate employee records.",
        "Learning and Development Manager designing and delivering training programs for leadership development, technical skills, and soft skills. Used LMS platforms (Moodle, Cornerstone). Measured training effectiveness through KPIs.",
        "Talent Acquisition Lead managing full-cycle recruiting for a 500-person tech company. Expert in employer branding, diversity hiring, and candidate experience. Reduced time-to-hire by 30% through process optimization.",
        "Compensation and Benefits Analyst conducting salary benchmarking, job evaluation, and benefits program administration. Proficient in Radford and Mercer compensation surveys. Ensured pay equity across the organization.",
        "HR Director overseeing HR strategy for a multinational company with 2000 employees. Expertise in mergers and acquisitions HR integration, executive compensation, and board-level HR reporting.",
        "Employee Engagement Specialist designing and implementing engagement surveys, recognition programs, and wellness initiatives. Improved eNPS score by 20 points. Skilled in data analysis and presenting insights to leadership.",
        "HR Operations Manager streamlining HR processes through HRIS implementation (BambooHR, SuccessFactors). Automated onboarding workflows, reduced manual HR tasks by 60%, and improved data accuracy.",
        "Industrial Relations Manager handling union negotiations, collective bargaining agreements, and labor dispute resolution. Deep knowledge of employment law, worker's rights, and regulatory compliance.",
        "Recruitment Coordinator managing job postings, interview scheduling, background checks, and offer letters for a high-volume hiring environment. Proficient in Greenhouse ATS and Microsoft Office Suite.",
        "HR Consultant providing advisory services to SMEs on HR policy creation, job design, performance appraisal systems, and HR audit. Delivered workshops on leadership, conflict resolution, and team building.",
        "Organizational Development Specialist designing competency frameworks, succession planning programs, and leadership pipelines. Facilitated team interventions and culture transformation initiatives.",
        "Payroll Manager overseeing payroll for 1000+ employees across multiple states. Expert in payroll compliance, tax filings, statutory deductions, and year-end processing. Proficient in ADP and SAP Payroll.",
    ],
    'Web Designing': [
        "Creative web designer with 4 years experience in UI/UX design, HTML5, CSS3, JavaScript, and responsive web design. Proficient in Figma, Adobe XD, and Photoshop. Created wireframes, prototypes, and visual designs.",
        "Front-end web developer and designer skilled in React.js, Vue.js, Tailwind CSS, and Bootstrap. Built pixel-perfect, accessible, and performant user interfaces. Experience in A/B testing and conversion rate optimization.",
        "UI/UX designer with a strong portfolio of mobile and web apps. Conducted user research, usability testing, and created design systems. Proficient in Figma, Sketch, and InVision. Background in graphic design.",
        "Web designer with expertise in WordPress, Elementor, WooCommerce, and custom theme development. Delivered 50+ client websites with SEO optimization. Skilled in Adobe Illustrator and brand identity design.",
        "Senior web designer leading product design at a SaaS startup. Established design systems, component libraries, and design tokens. Collaborated with product and engineering teams. Background in visual communication.",
        "Interaction designer specializing in animation and micro-interactions using CSS animations, GSAP, and Lottie. Built delightful user experiences with smooth transitions. Expert in Principle and ProtoPie for prototyping.",
        "Accessibility-focused web designer ensuring WCAG 2.1 AA compliance across all digital products. Conducted accessibility audits, screen reader testing, and keyboard navigation optimization. Advocate for inclusive design.",
        "E-commerce UX designer specializing in conversion rate optimization, checkout flow design, and product page layouts. Increased client revenue by 35% through data-driven design improvements.",
        "Web designer and front-end developer building JAMstack websites with Gatsby, Next.js, and Contentful CMS. Optimized Core Web Vitals scores and page load performance. Experienced in headless CMS architecture.",
        "Mobile-first web designer creating responsive layouts for complex data dashboards. Expert in CSS Grid, Flexbox, and SVG animations. Collaborated with backend teams to design REST API response formats.",
        "Visual designer with expertise in typography, color theory, and brand identity. Created complete brand guidelines, icon sets, and illustration libraries for tech startups. Tools: Figma, Illustrator, After Effects.",
        "Web designer specializing in landing page design and growth hacking. Created high-converting landing pages with clear CTAs. Ran A/B tests using Optimizely and analyzed user behavior with Hotjar.",
        "UX researcher and designer conducting user interviews, card sorting, and tree testing to inform design decisions. Created journey maps, personas, and service blueprints. Expert in Dovetail and Lookback for research.",
        "3D web designer creating immersive web experiences using Three.js, WebGL, and Spline. Built product configurators and interactive 3D product showcases for e-commerce. Background in 3D modeling with Blender.",
        "Web designer turned design lead managing a team of 4 designers. Introduced design sprints, critique sessions, and QA checklists. Improved design-to-development handoff with detailed Figma specifications.",
    ],
    'Mechanical Engineer': [
        "Mechanical engineer with 5 years experience in product design, CAD modeling using SolidWorks and AutoCAD, and FEA analysis. Worked in automotive industry on engine components. PMP certified project manager.",
        "Manufacturing engineer experienced in CNC machining, process optimization, lean manufacturing, and Six Sigma. Reduced production costs by 20%. Skilled in GD&T, quality control, and ISO 9001 compliance.",
        "HVAC engineer with expertise in heating, ventilation, and air conditioning system design. Used AutoCAD MEP and Revit MEP. Managed installation projects and ensured compliance with ASHRAE standards.",
        "Mechanical design engineer specializing in aerospace components. Experience with CATIA V5, ANSYS simulation, and composite materials. Familiar with FAA regulations and AS9100 quality standards.",
        "Thermal and fluids engineer with expertise in CFD simulations using ANSYS Fluent and COMSOL. Optimized heat exchanger designs and fluid flow systems. Published research on turbulence modeling.",
        "Robotics engineer designing mechanical systems for industrial robots. Expert in kinematics, actuator selection, and mechanism design. Proficient in SOLIDWORKS, MATLAB/Simulink, and PLC programming.",
        "Maintenance engineer managing preventive and corrective maintenance programs for heavy machinery. Expert in reliability-centered maintenance (RCM), CMMS systems, and root cause analysis using 5-why methodology.",
        "Automotive engineer with experience in engine calibration, powertrain development, and emissions testing. Proficient in MATLAB/Simulink, CANape, and INCA. Familiar with AUTOSAR and functional safety (ISO 26262).",
        "Structural engineer performing stress analysis and fatigue life calculations for pressure vessels and pipelines. Expert in ASME standards, weld analysis, and non-destructive testing (NDT). Oil and gas domain.",
        "Product development engineer taking products from concept to production. Expert in DFM/DFA principles, tolerance analysis, prototyping, and supplier qualification. Managed 10 product launches. PMP certified.",
        "Piping engineer designing process piping systems for chemical plants. Proficient in PDMS, Caesar II stress analysis, and P&ID drawings. Familiar with ASME B31.3, pressure relief valve sizing, and hazop studies.",
        "Renewable energy engineer designing wind turbine components and solar mounting structures. Expert in fatigue analysis, IEC standards, and energy yield assessments. Contributed to 500MW of renewable installations.",
        "Mechatronics engineer integrating mechanical, electronic, and software systems for automation solutions. Expert in PLC programming (Siemens, Allen Bradley), servo drives, and HMI design.",
        "Quality assurance engineer implementing QMS per IATF 16949 in automotive supply chain. Expert in FMEA, control plans, SPC, and MSA. Led supplier audits and managed corrective action processes.",
        "Manufacturing process engineer designing assembly lines and production workflows for consumer electronics. Used Arena simulation software for capacity planning and lean tools for waste elimination.",
    ],
    'Sales': [
        "Sales executive with 6 years B2B sales experience in technology sector. Consistently exceeded quarterly targets by 30%. Skilled in Salesforce CRM, cold calling, pipeline management, and contract negotiation.",
        "Regional sales manager leading a team of 15 sales representatives. Developed territory strategies and coaching programs. Grew regional revenue by 45% YoY. Expertise in enterprise software sales.",
        "Inside sales specialist proficient in consultative selling, lead qualification, and product demonstrations. Managed 200+ accounts. Used HubSpot CRM and outbound tools. Excellent communication skills.",
        "Retail sales associate with customer service and product knowledge in consumer electronics. Achieved top seller award 3 months in a row. Trained new team members. Strong in upselling and cross-selling.",
        "Business development manager experienced in identifying new market opportunities, building partnerships, and closing deals. Managed key accounts worth $5M annually. Strong negotiation and presentation skills.",
        "Enterprise account executive selling SaaS solutions to Fortune 500 companies. Expert in multi-stakeholder deals, RFP responses, and long sales cycles. Closed $10M+ in ARR. MEDDIC sales methodology.",
        "Sales engineer providing technical pre-sales support for industrial automation products. Conducted product demos, POCs, and technical proposal writing. Bridge between sales team and engineering.",
        "Channel sales manager building and managing reseller and partner networks across Southeast Asia. Expertise in partner enablement, MDF management, and channel conflict resolution. Grew channel revenue 60%.",
        "Pharmaceutical sales representative promoting prescription drugs to physicians and hospitals. Strong medical knowledge, regulatory compliance, and relationship building skills. Exceeded sales targets 4 consecutive years.",
        "Inside sales team lead coaching 10 SDRs on prospecting, objection handling, and CRM hygiene. Implemented cadence sequences using Outreach.io. Improved team's meeting booking rate from 8% to 15%.",
        "Real estate sales consultant with expertise in residential property transactions, market analysis, and buyer consultation. Top performing agent, $20M in transactions annually. Licensed realtor, expert negotiator.",
        "Solution sales specialist in cybersecurity products. Skilled in identifying customer pain points, delivering ROI-focused presentations, and navigating complex procurement processes. CISSP and security domain knowledge.",
        "E-commerce sales manager growing online revenue through marketplace optimization (Amazon, Shopify), digital advertising, and customer retention programs. Expert in analytics, pricing strategy, and inventory management.",
        "Sales operations analyst optimizing CRM processes, building sales dashboards, and conducting territory and quota planning. Proficient in Salesforce, Tableau, and SQL. Improved sales forecast accuracy by 25%.",
        "Field sales representative covering rural territories for FMCG company. Built distributor relationships, conducted product launches, and ensured shelf visibility. Managed 300+ outlets and exceeded volume targets.",
    ],
    'DevOps Engineer': [
        "DevOps engineer with 4 years experience in CI/CD pipelines, Docker, Kubernetes, and infrastructure as code using Terraform and Ansible. Managed AWS cloud infrastructure. Reduced deployment time by 60%.",
        "Site reliability engineer ensuring 99.99% uptime for microservices. Expertise in Prometheus, Grafana, ELK stack monitoring. Automated incident response. Deep knowledge of Linux, networking, and security.",
        "Cloud engineer with AWS and GCP certifications. Designed and deployed scalable architectures, implemented auto-scaling, load balancing, and disaster recovery. Used CloudFormation and Pulumi.",
        "DevOps specialist with Jenkins, GitHub Actions, GitLab CI expertise. Built automated testing, security scanning, and deployment pipelines. Implemented GitOps practices with ArgoCD and Flux.",
        "Kubernetes administrator managing multi-cluster production environments. Expertise in Helm, service mesh (Istio), and network policies. Contributed to CNCF open source projects. CKA certified.",
        "Platform engineer building internal developer platforms to improve engineering productivity. Expert in Backstage, Crossplane, and developer portal development. Championed DevEx improvements reducing onboarding time.",
        "Azure DevOps engineer managing CI/CD pipelines, Azure Kubernetes Service, and Azure DevOps boards. Expertise in ARM templates, Azure Policy, and cost optimization. Microsoft Azure certified solutions architect.",
        "Security DevOps (DevSecOps) engineer integrating security scanning (SAST, DAST, SCA) into CI/CD pipelines. Used Snyk, SonarQube, and Trivy. Implemented secrets management with HashiCorp Vault.",
        "Linux systems administrator with 8 years experience in server management, shell scripting, and performance tuning. Migrated on-premises infrastructure to AWS. Expert in Bash, Python automation, and cron scheduling.",
        "Database reliability engineer managing PostgreSQL and MySQL clusters at scale. Expert in replication, backup strategies, performance tuning, and connection pooling with PgBouncer. AWS RDS and Aurora experience.",
        "Network engineer and DevOps practitioner automating network configuration with Ansible and Python. Expert in BGP, OSPF, VLANs, and SD-WAN. Cisco CCNP certified with cloud networking experience.",
        "Infrastructure engineer building multi-region cloud architectures for high availability. Expert in AWS VPC design, Transit Gateway, Route 53 failover, and cross-region replication strategies.",
        "Observability engineer building comprehensive monitoring stacks with Prometheus, Loki, Tempo, and Grafana. Implemented distributed tracing, SLO/SLA dashboards, and on-call alert routing with PagerDuty.",
        "Build and release engineer managing software release cycles, version control workflows, and artifact management with Nexus and JFrog Artifactory. Expert in semantic versioning and release automation.",
        "FinOps engineer optimizing cloud costs across AWS, Azure, and GCP. Used CloudHealth, Spot.io, and AWS Cost Explorer. Achieved 35% reduction in cloud spend through reserved instances and rightsizing.",
    ],
    'Network Security Engineer': [
        "Network security engineer with 5 years experience in firewall configuration (Palo Alto, Cisco ASA), intrusion detection systems, VPN setup, and vulnerability assessments. CISSP and CEH certified.",
        "Cybersecurity analyst experienced in SIEM tools (Splunk, QRadar), threat intelligence, incident response, and penetration testing. Conducted security audits and ensured SOC 2 compliance.",
        "Network engineer specializing in routing protocols (BGP, OSPF), switching, and network design. Managed enterprise LAN/WAN infrastructure. CCNP certified. Experience with SD-WAN solutions.",
        "Information security engineer with expertise in identity and access management (IAM), PKI, and zero-trust architecture. Implemented LDAP/AD integration and multi-factor authentication systems.",
        "Security operations center analyst monitoring and responding to security events 24/7. Proficient in threat hunting, malware analysis, and forensics. SANS GIAC certified. Wrote security playbooks.",
        "Penetration tester conducting web application, network, and mobile security assessments. Expert in Metasploit, Burp Suite, Nmap, and custom exploit development. OSCP certified. Bug bounty hunter.",
        "Cloud security engineer securing AWS and Azure environments. Expert in IAM policies, Security Groups, CloudTrail, GuardDuty, and compliance frameworks (CIS, NIST, PCI-DSS). Cloud Security Alliance member.",
        "Incident response specialist leading forensic investigations of security breaches. Expert in memory forensics, disk imaging, log analysis, and threat actor attribution. Authored incident response playbooks.",
        "Application security engineer conducting code reviews, threat modeling, and SAST/DAST scanning. Expertise in OWASP Top 10, secure SDLC integration, and developer security training. CEH and CSSLP certified.",
        "Network architect designing secure enterprise networks with zero-trust segmentation. Expert in micro-segmentation, encrypted DNS, SSL inspection, and network access control (NAC). CCIE Security certified.",
        "Red team operator conducting adversarial simulations against enterprise environments. Expertise in phishing campaigns, Active Directory attacks, lateral movement, and C2 frameworks. CRTO certified.",
        "Security compliance analyst managing ISO 27001, SOC 2 Type II, and GDPR compliance programs. Conducted risk assessments, vendor due diligence, and audit preparation. Developed security awareness training.",
        "Vulnerability management engineer running enterprise vulnerability scanning with Qualys and Nessus. Prioritized remediation using CVSS scores and threat intelligence. Reduced critical vulnerability count by 70%.",
        "Wireless security engineer conducting Wi-Fi security assessments and designing secure WLAN architectures. Expert in WPA3, 802.1X, and rogue AP detection. Familiar with IoT security and OT network protection.",
        "DLP and data security specialist implementing data loss prevention policies, data classification, and encryption strategies. Expert in Microsoft Purview, Symantec DLP, and insider threat detection programs.",
    ],
    'Business Analyst': [
        "Business analyst with 5 years experience gathering requirements, creating BRDs, process mapping, and stakeholder management. Proficient in JIRA, Confluence, and Visio. CBAP certified.",
        "Agile business analyst working in scrum teams writing user stories, acceptance criteria, and sprint planning. Experience in digital transformation projects. Strong SQL skills for data analysis.",
        "Financial business analyst experienced in ERP implementations (SAP, Oracle), financial modeling, budgeting, and reporting. Used Power BI and Tableau for dashboards. CFA Level 1 candidate.",
        "IT business analyst bridging technical teams and business stakeholders. Conducted gap analysis, UAT coordination, and change management. Experience in banking and insurance domains.",
        "Product analyst using data to drive product decisions. Proficient in Google Analytics, Mixpanel, SQL, and Python for analysis. Created A/B testing frameworks and defined product KPIs.",
        "Senior business analyst leading requirements workshops for large-scale system integrations. Expert in BPMN 2.0 process modeling, use case development, and functional specifications. PMP and CBAP dual certified.",
        "E-commerce business analyst analyzing customer funnel, conversion rates, and cohort retention. Proficient in SQL, Excel, and Tableau. Partnered with marketing and product teams to drive revenue growth.",
        "Healthcare business analyst managing EMR/EHR implementations (Epic, Cerner). Expert in HL7, FHIR standards, clinical workflow analysis, and regulatory compliance (HIPAA). Improved patient data accuracy by 40%.",
        "Business analyst specializing in supply chain and logistics optimization. Mapped order-to-delivery processes, identified bottlenecks, and implemented ERP solutions (SAP, Oracle SCM). Reduced costs by $2M annually.",
        "Data business analyst combining traditional BA skills with advanced analytics. Built predictive models, automated reports, and self-service dashboards. Proficient in Python, SQL, Tableau, and PowerBI.",
        "Business systems analyst managing CRM implementations (Salesforce, Dynamics 365). Gathered requirements from sales and marketing teams, configured workflows, and delivered user training. Certified Salesforce Admin.",
        "Regulatory business analyst working in compliance-heavy environments (banking, pharma). Expert in regulatory change analysis, impact assessment, and policy documentation. Strong understanding of Basel III and MiFID II.",
        "Business analyst in retail domain analyzing sales trends, inventory levels, and category performance. Built dashboards in Power BI and performed root cause analysis on revenue gaps. Advanced Excel user.",
        "Digital transformation business analyst helping enterprises migrate from legacy systems to cloud. Expert in as-is/to-be analysis, business case development, and vendor selection. Managed stakeholders at C-suite level.",
        "Junior business analyst with 2 years experience supporting senior BAs in requirements elicitation, documentation, and testing. Familiar with JIRA, Confluence, and basic SQL. Completed IIBA ECBA certification.",
    ],
    'Accountant': [
        "Chartered accountant with 6 years experience in financial reporting, audit, tax compliance, and IFRS accounting. Managed accounts for mid-sized manufacturing companies. Expert in SAP FI module and MS Excel.",
        "Management accountant preparing monthly management accounts, budgets, forecasts, and variance analysis reports. Proficient in Oracle Financials and Power BI for financial dashboards. CIMA qualified.",
        "Tax accountant specializing in corporate tax returns, VAT filings, and transfer pricing documentation. Experience with international tax compliance across 15 countries. Proficient in ONESOURCE tax software.",
        "Audit senior conducting statutory audits for listed companies. Expert in risk-based audit methodology, ISA standards, and audit documentation. Used IDEA and TeamMate for audit management. ICAEW qualified.",
        "Financial controller overseeing all accounting operations for a $50M revenue company. Responsibilities include month-end close, consolidation, treasury management, and internal controls. CPA with Big 4 background.",
        "Accounts payable specialist processing high-volume invoices, reconciling vendor statements, and managing payment runs. Expert in SAP AP module, 3-way matching, and resolving invoice disputes.",
        "Accounts receivable accountant managing customer billing, collections, and credit risk assessment. Reduced DSO by 15 days through improved collection processes. Proficient in NetSuite and Salesforce CRM integration.",
        "Cost accountant performing product costing, standard costing variance analysis, and overhead absorption calculations. Supported pricing decisions with detailed cost models. Manufacturing domain expertise.",
        "Payroll accountant managing monthly payroll for 500+ employees across multiple jurisdictions. Expert in statutory deductions, year-end tax filings, and payroll reconciliation. Proficient in ADP and QuickBooks.",
        "Financial analyst preparing financial models, investment appraisals, and due diligence reports for M&A transactions. Expert in DCF valuation, LBO modeling, and scenario analysis. CFA Level 2 candidate.",
        "Treasury analyst managing cash flow forecasting, FX hedging, and liquidity management. Proficient in Treasury Management Systems (Kyriba), Bloomberg, and derivative instruments.",
        "Internal auditor evaluating internal controls, identifying compliance risks, and recommending process improvements. Conducted operational, financial, and IT audits. CIA certified with expertise in SOX compliance.",
        "Bookkeeper maintaining general ledger entries, bank reconciliations, and trial balance for small businesses. Proficient in Xero, QuickBooks, and Sage. Prepared quarterly VAT returns and statutory accounts.",
        "Forensic accountant investigating financial fraud, embezzlement, and asset misappropriation. Expert in transaction analysis, document review, and expert witness reporting. CFE (Certified Fraud Examiner) certified.",
        "Group financial reporting accountant preparing consolidated financial statements under IFRS. Expert in intercompany eliminations, goodwill impairment, and equity accounting for associates and JVs.",
    ],
    'Testing': [
        "QA engineer with 4 years experience in manual and automated testing. Proficient in Selenium, Python, and TestNG. Developed test plans, test cases, and defect management using JIRA. Agile/Scrum environment.",
        "Automation test engineer building robust test frameworks using Selenium WebDriver, Java, and Maven. Implemented Page Object Model design pattern. Integrated tests into Jenkins CI pipeline. ISTQB certified.",
        "Performance test engineer using JMeter and Gatling to conduct load, stress, and soak tests. Identified bottlenecks in API and database layers. Proficient in performance monitoring with APM tools.",
        "Mobile test engineer testing iOS and Android applications using Appium and Espresso. Expert in device farm testing (BrowserStack, Sauce Labs), accessibility testing, and beta testing programs.",
        "API test engineer using Postman, RestAssured, and SoapUI to test RESTful and SOAP services. Automated API test suites integrated into CI/CD pipelines. Expert in JSON schema validation and contract testing.",
        "Security test engineer performing application security testing, OWASP vulnerability checks, and penetration testing on web applications. Used Burp Suite, OWASP ZAP, and manual techniques. GWAPT certified.",
        "Test lead managing a QA team of 8, defining testing strategy, estimating effort, and reporting test metrics to stakeholders. Introduced shift-left testing practices and reduced escaped defects by 50%.",
        "ETL test engineer validating data pipelines, data warehouse loads, and reporting layers. Expert in SQL-based data validation, data profiling, and metadata testing. Experience with Informatica and Talend.",
        "Agile QA analyst embedded in scrum teams, writing acceptance criteria, conducting exploratory testing, and providing rapid feedback. Expert in BDD with Cucumber and Gherkin syntax. Strong collaboration skills.",
        "Test automation architect designing scalable test frameworks using Playwright, Cypress, and TypeScript. Introduced testing best practices, code reviews for test code, and test data management strategies.",
        "Usability tester conducting moderated and unmoderated user testing sessions, tree testing, and heuristic evaluations. Worked closely with UX designers to validate designs before development. Expert in UserTesting platform.",
        "Game QA tester performing functional, regression, and compatibility testing for AAA mobile games. Expert in bug reporting, reproduction steps, and regression test cycles. Familiar with Unreal Engine and Unity.",
        "Embedded software tester validating firmware for IoT devices and automotive ECUs. Expert in hardware-in-the-loop testing, CAN bus analysis, and requirements-based testing using DOORS. Automotive SPICE experience.",
        "Test data management specialist creating and maintaining synthetic test data for complex enterprise systems. Expert in data masking, data subsetting, and test environment management with Enov8.",
        "Continuous testing evangelist integrating testing throughout the DevOps lifecycle. Implemented test parallelization, flaky test detection, and quality gates in deployment pipelines. Reduced regression test time from 4h to 30min.",
    ],
}


class DataLoader:
    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.data_path):
            print(f"  Loading dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            df = self._standardize_columns(df)
        else:
            print("  Dataset file not found. Using built-in sample data (150 resumes, 10 categories).")
            print("  Run download_dataset.py to get the full Kaggle dataset for better accuracy.")
            df = self._generate_sample_data()

        df = self._clean_dataframe(df)
        print(f"  Loaded {len(df)} resumes across {df['Category'].nunique()} categories.")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ('category', 'label', 'job_category', 'job category'):
                col_map[col] = 'Category'
            elif lower in ('resume', 'resume_text', 'text', 'content'):
                col_map[col] = 'Resume'
        return df.rename(columns=col_map)[['Category', 'Resume']]

    def _generate_sample_data(self) -> pd.DataFrame:
        rows = []
        for category, texts in SAMPLE_RESUMES.items():
            for text in texts:
                rows.append({'Category': category, 'Resume': text})
        df = pd.DataFrame(rows)
        np.random.seed(42)
        return df.sample(frac=1).reset_index(drop=True)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['Category', 'Resume'])
        df['Category'] = df['Category'].str.strip()
        df['Resume'] = df['Resume'].astype(str).str.strip()
        df = df[df['Resume'].str.len() > 20]
        return df.reset_index(drop=True)

    def get_stats(self, df: pd.DataFrame) -> dict:
        return {
            'total_resumes': len(df),
            'num_categories': df['Category'].nunique(),
            'categories': df['Category'].value_counts().to_dict(),
            'avg_resume_length': int(df['Resume'].str.len().mean()),
            'min_resume_length': int(df['Resume'].str.len().min()),
            'max_resume_length': int(df['Resume'].str.len().max()),
        }
